import os
import logging
import shutil
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from config import Config as cfg
from dataloader_utils import create_training_dataset
from losses import select_G_loss_fn, select_D_loss_fn, create_loss_vars
from metrics.metrics_utils import setup_metrics
from networks import ModelConfig, Generator, Discriminator
# Utils imports
from checkpoint_utils import save_model, load_model, save_optimizer_loss_scale, load_optimizer_loss_scale,\
    remove_old_models
from utils import should_use_fp16,\
    create_images_dir_path, create_images_grid_title,\
    format_time, is_last_step, should_write_summary,\
    load_images_paths, fast_save_grid
from utils import NHWC_FORMAT, NCHW_FORMAT,\
    DEFAULT_MODE, TRAIN_MODE, INFERENCE_MODE, BENCHMARK_MODE,\
    GENERATOR_NAME, DISCRIMINATOR_NAME, SMOOTH_POSTFIX, OPTIMIZER_POSTFIX,\
    CACHE_DIR, DATASET_CACHE_DIR, MODELS_DIR, TF_LOGS_DIR
from tf_utils import generate_latents, is_finite_grad, update_loss_scale, should_update_loss_scale, \
    trace_vars, maybe_show_vars_stats, maybe_show_grads_stat, get_gpu_memory_usage,\
    maybe_scale_loss, maybe_unscale_grads, set_optimizer_iters, set_tf_logging,\
    smooth_model_weights, convert_outputs_to_images, run_model_on_batches
from tf_utils import DEFAULT_DATA_FORMAT, toNCHW_AXIS, toNHWC_AXIS, MAX_LOSS_SCALE


set_tf_logging(debug_mode=False)


CPU_DEVICE = '/CPU:0'

D_KEY  = 'D'
G_KEY  = 'G'
GS_KEY = 'Gs'

FIRST_STEP_COND_KEY          = 'first_step_cond'
LAST_STEP_COND_KEY           = 'last_step_cond'
STAGE_IMAGES_KEY             = 'stage_images'
TRAINING_FINISHED_IMAGES_KEY = 'training_finished_images'
WRITE_SCALARS_SUMMARY_KEY    = 'write_scalars_summary'
WRITE_HISTS_SUMMARY_KEY      = 'write_hists_summary'
WRITE_LOSS_SCALE_SUMMARY_KEY = 'write_loss_scale_summary'
RUN_METRICS_KEY              = 'run_metrics'
SAVE_MODELS_KEY              = 'save_models'
SAVE_VALID_IMAGES_KEY        = 'save_valid_images'
SMOOTH_G_WEIGHTS_KEY         = 'smooth_G_weights'
RESET_LOSS_SCALE_STATS_KEY   = 'reset_loss_scale_stats'
EVALUATE_G_REG_KEY           = 'evaluate_G_reg'
EVALUATE_D_REG_KEY           = 'evaluate_D_reg'


def tf_bool(x):
    return tf.convert_to_tensor(x, dtype=tf.bool)


class StyleGAN2(ModelConfig):

    def __init__(self, config, mode=DEFAULT_MODE, images_paths=None):
        super(StyleGAN2, self).__init__(config)

        # TODO: think about removing inference mode (causes errors during graphs tracing)
        if mode in [TRAIN_MODE, BENCHMARK_MODE, INFERENCE_MODE]:
            # Training images and batches
            self.latent_size = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)

            # Smoothed generator
            self.use_Gs               = config.get(cfg.USE_G_SMOOTHING, cfg.DEFAULT_USE_G_SMOOTHING)
            self.use_gpu_for_Gs       = config.get(cfg.USE_GPU_FOR_GS, cfg.DEFAULT_USE_GPU_FOR_GS)
            self.Gs_beta              = config.get(cfg.G_SMOOTHING_BETA, cfg.DEFAULT_G_SMOOTHING_BETA)
            self.Gs_beta_kimgs        = config.get(cfg.G_SMOOTHING_BETA_KIMAGES, cfg.DEFAULT_G_SMOOTHING_BETA_KIMAGES)

            # Dataset
            self.dataset_hw_ratio      = config.get(cfg.DATASET_HW_RATIO, cfg.DEFAULT_DATASET_HW_RATIO)
            self.dataset_max_cache_res = config.get(cfg.DATASET_MAX_CACHE_RES, cfg.DEFAULT_DATASET_MAX_CACHE_RES)
            if images_paths is None:
                images_paths = load_images_paths(config)
            # These options are used for metrics
            self.dataset_params = {
                'fpaths'              : images_paths,
                'hw_ratio'            : self.dataset_hw_ratio,
                'mirror_augment'      : config.get(cfg.MIRROR_AUGMENT, cfg.DEFAULT_MIRROR_AUGMENT),
                'shuffle_dataset'     : config.get(cfg.SHUFFLE_DATASET, cfg.DEFAULT_SHUFFLE_DATASET),
                'data_format'         : self.data_format,
                'use_fp16'            : self.use_mixed_precision,
                'n_parallel_calls'    : config.get(cfg.DATASET_N_PARALLEL_CALLS, cfg.DEFAULT_DATASET_N_PARALLEL_CALLS),
                'n_prefetched_batches': config.get(cfg.DATASET_N_PREFETCHED_BATCHES, cfg.DEFAULT_DATASET_N_PREFETCHED_BATCHES)
            }

            # Losses
            self.G_loss_fn_name      = config.get(cfg.G_LOSS_FN, cfg.DEFAULT_G_LOSS_FN)
            self.D_loss_fn_name      = config.get(cfg.D_LOSS_FN, cfg.DEFAULT_D_LOSS_FN)
            self.G_loss_fn           = select_G_loss_fn(self.G_loss_fn_name, use_xla=False)
            self.D_loss_fn           = select_D_loss_fn(self.D_loss_fn_name, use_xla=False)
            self.G_loss_params       = config.get(cfg.G_LOSS_FN_PARAMS, cfg.DEFAULT_G_LOSS_FN_PARAMS)
            self.D_loss_params       = config.get(cfg.D_LOSS_FN_PARAMS, cfg.DEFAULT_D_LOSS_FN_PARAMS)
            self.lazy_regularization = config.get(cfg.LAZY_REGULARIZATION, cfg.DEFAULT_LAZY_REGULARIZATION)
            self.G_reg_interval      = config.get(cfg.G_REG_INTERVAL, cfg.DEFAULT_G_REG_INTERVAL)
            self.D_reg_interval      = config.get(cfg.D_REG_INTERVAL, cfg.DEFAULT_D_REG_INTERVAL)
            self.loss_vars           = create_loss_vars()

            # Optimizers options
            self.G_learning_rate = config.get(cfg.G_LEARNING_RATE, cfg.DEFAULT_G_LEARNING_RATE)
            self.D_learning_rate = config.get(cfg.D_LEARNING_RATE, cfg.DEFAULT_D_LEARNING_RATE)
            self.beta1 = config.get(cfg.ADAM_BETA1, cfg.DEFAULT_ADAM_BETA1)
            self.beta2 = config.get(cfg.ADAM_BETA2, cfg.DEFAULT_ADAM_BETA2)
            self.loss_scale_cycle_length = 1000 # Number of iterations for loss scale cycle length. Used for dealing with small loss scales

            # Valid images options
            self.valid_grid_nrows = config.get(cfg.VALID_GRID_NROWS, cfg.DEFAULT_VALID_GRID_NROWS)
            self.valid_grid_ncols = config.get(cfg.VALID_GRID_NCOLS, cfg.DEFAULT_VALID_GRID_NCOLS)
            self.valid_grid_padding = 2
            self.min_target_single_image_size = config.get(cfg.VALID_MIN_TARGET_SINGLE_IMAGE_SIZE, cfg.DEFAULT_VALID_MIN_TARGET_SINGLE_IMAGE_SIZE)
            if self.min_target_single_image_size < 0:
               self.min_target_single_image_size = max(2 ** (self.resolution_log2 - 1), 2 ** 7)
            self.max_png_res = config.get(cfg.VALID_MAX_PNG_RES, cfg.DEFAULT_VALID_MAX_PNG_RES)

            # Summaries
            self.model_name            =            config[cfg.MODEL_NAME]
            self.metrics               =            config.get(cfg.METRICS_DICT, cfg.DEFAULT_METRICS_DICT)
            self.storage_path          =            config.get(cfg.STORAGE_PATH, cfg.DEFAULT_STORAGE_PATH)
            self.max_models_to_keep    =            config.get(cfg.MAX_MODELS_TO_KEEP, cfg.DEFAULT_MAX_MODELS_TO_KEEP)
            self.run_metrics_every     = int(1000 * config.get(cfg.RUN_METRICS_EVERY_KIMAGES, cfg.DEFAULT_RUN_METRICS_EVERY_KIMAGES))
            self.summary_scalars_every = int(1000 * config.get(cfg.SUMMARY_SCALARS_EVERY_KIMAGES, cfg.DEFAULT_SUMMARY_SCALARS_EVERY_KIMAGES))
            self.summary_hists_every   = int(1000 * config.get(cfg.SUMMARY_HISTS_EVERY_KIMAGES, cfg.DEFAULT_SUMMARY_HISTS_EVERY_KIMAGES))
            self.save_model_every      = int(1000 * config.get(cfg.SAVE_MODEL_EVERY_KIMAGES, cfg.DEFAULT_SAVE_MODEL_EVERY_KIMAGES))
            self.save_images_every     = int(1000 * config.get(cfg.SAVE_IMAGES_EVERY_KIMAGES, cfg.DEFAULT_SAVE_IMAGES_EVERY_KIMAGES))
            self.logs_path             = os.path.join(MODELS_DIR, self.model_name, TF_LOGS_DIR)
            self.summary_writer        = tf.summary.create_file_writer(self.logs_path)
            self.valid_latents         = self.initialize_valid_latents()

        self.G_object = Generator(config)
        self.D_object = Discriminator(config)
        # Maybe create smoothed generator
        if self.use_Gs:
            Gs_config = config
            self.Gs_valid_latents = self.valid_latents
            self.Gs_device = '/GPU:0' if self.use_gpu_for_Gs else CPU_DEVICE
            if not self.use_gpu_for_Gs:
                Gs_config[cfg.DATA_FORMAT] = NHWC_FORMAT
                self.Gs_valid_latents = tf.transpose(self.valid_latents, toNHWC_AXIS)
            self.Gs_object = Generator(Gs_config)

        self.initialize_models()
        if mode == INFERENCE_MODE:
            print('Ready for inference')
        else:
            self.setup_Gs_beta()
            self.create_images_dataset()
            self.metrics_objects = setup_metrics(2 ** self.resolution_log2,
                                                 hw_ratio=self.dataset_hw_ratio,
                                                 dataset_params=self.dataset_params,
                                                 use_fp16=self.use_mixed_precision,
                                                 use_xla=self.use_xla,
                                                 model_name=self.model_name,
                                                 metrics=self.metrics,
                                                 benchmark_mode=(mode == BENCHMARK_MODE))
            if mode == BENCHMARK_MODE:
                self.initialize_optimizers(benchmark=True)
            elif mode == TRAIN_MODE:
                self.initialize_optimizers()

    def initialize_valid_latents(self):
        latents_dir = os.path.join(MODELS_DIR, self.model_name, CACHE_DIR)
        latents_path = os.path.join(latents_dir, 'latents.npy')
        if os.path.exists(latents_path):
            latents = tf.constant(np.load(latents_path, allow_pickle=False))
            logging.info('Loaded valid latents from file')
        else:
            os.makedirs(latents_dir, exist_ok=True)
            latents = self.generate_latents(self.valid_grid_nrows * self.valid_grid_ncols)
            np.save(latents_path, latents.numpy(), allow_pickle=False)
            logging.info('Valid latents not found. Created and saved new samples')
        return latents

    def initialize_models(self):
        if self.use_mixed_precision:
            logging.info(f'Start fp16 resolution: {self.start_fp16_resolution_log2}')
        self.G_object.initialize_G_model()
        self.D_object.initialize_D_model()
        if self.use_Gs:
            with tf.device(self.Gs_device):
                self.Gs_object.initialize_G_model(plot_model=False)
                G_model = self.G_object.create_G_model()
                Gs_model = self.Gs_object.create_G_model()
                Gs_model.set_weights(G_model.get_weights())

    def setup_Gs_beta(self):
        if self.Gs_beta is None:
            beta = tf.constant(0.5 ** (self.batch_size / (1000.0 * self.Gs_beta_kimgs)), dtype='float32')
        else:
            beta = tf.constant(self.Gs_beta, dtype='float32')
        logging.info(f'Gs beta: {beta}')
        self.Gs_beta = beta

    def trace_graphs(self):
        self.G_object.trace_G_graph(self.summary_writer, self.logs_path)
        self.D_object.trace_D_graph(self.summary_writer, self.logs_path)

        logging.info('\nGenerator network:\n')
        self.G_object.G_mapping.summary(print_fn=logging.info)
        self.G_object.G_synthesis.summary(print_fn=logging.info)
        self.G_object.G_model.summary(print_fn=logging.info)

        logging.info('\nDiscriminator network:\n')
        self.D_object.D_model.summary(print_fn=logging.info)

    def initialize_G_optimizer(self, benchmark: bool = False):
        c = self.G_reg_interval / (self.G_reg_interval + 1) if self.lazy_regularization else 1
        self.G_optimizer = tf.keras.optimizers.Adam(
            learning_rate=c * self.G_learning_rate,
            beta_1=self.beta1 ** c,
            beta_2=self.beta2 ** c,
            epsilon=1e-8,
            name='G_Adam'
        )
        if self.use_mixed_precision:
            initial_scale, dynamic = self.get_optimizer_initial_loss_scale(GENERATOR_NAME, benchmark)
            self.G_optimizer = LossScaleOptimizer(self.G_optimizer, dynamic=dynamic, initial_scale=initial_scale)
        self.G_optimizer.use_mixed_precision = self.use_mixed_precision

    def initialize_D_optimizer(self, benchmark: bool = False):
        c = self.D_reg_interval / (self.D_reg_interval + 1) if self.lazy_regularization else 1
        self.D_optimizer = tf.keras.optimizers.Adam(
            learning_rate=c * self.D_learning_rate,
            beta_1=self.beta1 ** c,
            beta_2=self.beta2 ** c,
            epsilon=1e-8,
            name='D_Adam'
        )
        if self.use_mixed_precision:
            initial_scale, dynamic = self.get_optimizer_initial_loss_scale(DISCRIMINATOR_NAME, benchmark)
            self.D_optimizer = LossScaleOptimizer(self.D_optimizer, dynamic=dynamic, initial_scale=initial_scale)
        self.D_optimizer.use_mixed_precision = self.use_mixed_precision

    def get_optimizer_initial_loss_scale(self, model_type, benchmark: bool = False):
        if benchmark:
            # Use default values for loss scale optimizer
            return MAX_LOSS_SCALE, True

        use_fp16 = self.use_mixed_precision
        if not use_fp16:
            logging.info(f"Model doesn't use mixed precision, so constant loss scale for {model_type} optimizer is set to 1")
            return 1., False
        else:
            logging.info(f"Loss scale for {model_type} optimizer is set to {MAX_LOSS_SCALE}")
            return MAX_LOSS_SCALE, True

    def initialize_optimizers(self, benchmark: bool = False):
        start_time = time.time()
        logging.info('Initializing optimizers...')
        self.initialize_G_optimizer(benchmark)
        self.initialize_D_optimizer(benchmark)
        total_time = time.time() - start_time
        logging.info(f'Optimizers initialized in {total_time:.3f} seconds!')

    def restore_optimizers_state(self, step):
        logging.info(f'Restoring optimizer state for step={step}...')
        shared_kwargs = {
            'model_name': self.model_name,
            'step': step,
            'storage_path': self.storage_path
        }

        MODEL_ARG = 'model'
        MODEL_TYPE_ARG = 'model_type'

        D_optimizer_kwargs = {
            MODEL_ARG: self.D_optimizer._optimizer if self.use_mixed_precision else self.D_optimizer,
            MODEL_TYPE_ARG: DISCRIMINATOR_NAME + OPTIMIZER_POSTFIX,
            **shared_kwargs
        }
        G_optimizer_kwargs = {
            MODEL_ARG: self.G_optimizer._optimizer if self.use_mixed_precision else self.G_optimizer,
            MODEL_TYPE_ARG: GENERATOR_NAME + OPTIMIZER_POSTFIX,
            **shared_kwargs
        }

        if self.use_mixed_precision:
            self.D_optimizer._optimizer = load_model(**D_optimizer_kwargs)
            self.G_optimizer._optimizer = load_model(**G_optimizer_kwargs)
        else:
            self.D_optimizer = load_model(**D_optimizer_kwargs)
            self.G_optimizer = load_model(**G_optimizer_kwargs)

        set_optimizer_iters(self.D_optimizer, step)
        set_optimizer_iters(self.G_optimizer, step)

        D_iters = self.D_optimizer.iterations
        G_iters = self.G_optimizer.iterations
        logging.info(f'D opt iters: var is {D_iters.name} and value is {D_iters.numpy()}')
        logging.info(f'G opt iters: var is {G_iters.name} and value is {G_iters.numpy()}')

    def reset_loss_scale_states(self):
        # If loss scaly is lower than that, then try forcefully increasing it
        self.loss_scale_threshold = 8
        # Max number of updates during loss scale cycles
        self.G_cycle_left_updates = 5
        self.D_cycle_left_updates = 5

    def maybe_update_loss_scales(self):
        # G optimizer
        if self.G_cycle_left_updates > 0:
            if should_update_loss_scale(self.G_optimizer, self.loss_scale_threshold):
                update_loss_scale(self.G_optimizer, 'G')
                self.G_cycle_left_updates -= 1
        # D optimizer
        if self.D_cycle_left_updates > 0:
            if should_update_loss_scale(self.D_optimizer, self.loss_scale_threshold):
                update_loss_scale(self.D_optimizer, 'D')
                self.D_cycle_left_updates -= 1

    def create_images_dataset(self):
        start_time = time.time()
        logging.info(f'Initializing images dataset...')

        # No caching by default
        cache = False
        if self.dataset_max_cache_res is not None:
            if self.resolution_log2 <= self.dataset_max_cache_res:
                cache = os.path.join(self.storage_path or '', MODELS_DIR, self.model_name, DATASET_CACHE_DIR)
                os.makedirs(cache, exist_ok=True)

        res_kwargs = {'res': self.resolution_log2, 'batch_size': self.batch_size, 'cache': cache}
        self.images_dataset = iter(create_training_dataset(**{**res_kwargs, **self.dataset_params}))

        total_time = time.time() - start_time
        logging.info(f'Images dataset initialized in {total_time:.3f} seconds!')

    def create_models(self):
        # All models should be initialized before calling this function
        D_model = self.D_object.create_D_model()
        G_model = self.G_object.create_G_model()
        logging.info('Creating Gs model')
        if self.use_Gs:
            Gs_model = self.Gs_object.create_G_model()
            Gs_model.set_weights(G_model.get_weights())
        else:
            Gs_model = None

        # Log D model
        D_model.summary(print_fn=logging.info)
        # Log G model (mapping and synthesis networks)
        self.G_object.G_mapping.summary(print_fn=logging.info)
        self.G_object.G_synthesis.summary(print_fn=logging.info)
        # G_model.summary(print_fn=logging.info)

        logging.info('Models created!')

        return D_model, G_model, Gs_model

    def update_models_weights(self, models):
        self.D_object.save_D_weights_in_class(models[D_KEY])
        self.G_object.save_G_weights_in_class(models[G_KEY])
        if self.use_Gs:
            self.Gs_object.save_G_weights_in_class(models[GS_KEY])

    def load_models_from_class(self, models):
        D_model = self.D_object.load_D_weights_from_class(models[D_KEY])
        G_model = self.G_object.load_G_weights_from_class(models[G_KEY])
        if self.use_Gs:
            Gs_model = self.Gs_object.load_G_weights_from_class(models[GS_KEY])
        else:
            Gs_model = None
        return D_model, G_model, Gs_model

    def load_trained_models(self, models, step):
        logging.info(f'Loading models from step={step}...')

        D_model, G_model, Gs_model = models[D_KEY], models[G_KEY], models[GS_KEY]

        maybe_show_vars_stats(D_model.trainable_variables, 'D stats after init:')
        maybe_show_vars_stats(G_model.trainable_variables, 'G stats after init:')

        D_model = load_model(
            D_model, self.model_name, DISCRIMINATOR_NAME,
            step=step, storage_path=self.storage_path
        )
        G_model = load_model(
            G_model, self.model_name, GENERATOR_NAME,
            step=step, storage_path=self.storage_path
        )
        if Gs_model is not None:
            Gs_model = load_model(
                Gs_model, self.model_name, GENERATOR_NAME + SMOOTH_POSTFIX,
                step=step, storage_path=self.storage_path
            )

        maybe_show_vars_stats(D_model.trainable_variables, '\nD stats after loading:')
        maybe_show_vars_stats(G_model.trainable_variables, '\nG stats after loading:')

        logging.info(f'Loaded model weights from step={step}')
        return D_model, G_model, Gs_model

    def save_models(self, models, step):
        D_model, G_model, Gs_model = models[D_KEY], models[G_KEY], models[GS_KEY]
        shared_kwargs = {
            'model_name': self.model_name,
            'step': step,
            'storage_path': self.storage_path
        }
        save_model(model=D_model, model_type=DISCRIMINATOR_NAME, **shared_kwargs)
        save_model(model=G_model, model_type=GENERATOR_NAME, **shared_kwargs)
        if Gs_model is not None:
            save_model(model=Gs_model, model_type=GENERATOR_NAME + SMOOTH_POSTFIX, **shared_kwargs)

    def save_optimizers_weights(self, step=None):
        shared_kwargs = {
            'model_name': self.model_name,
            'step': step,
            'storage_path': self.storage_path
        }

        MODEL_ARG = 'model'
        MODEL_TYPE_ARG = 'model_type'

        D_model_type = DISCRIMINATOR_NAME + OPTIMIZER_POSTFIX
        G_model_type = GENERATOR_NAME + OPTIMIZER_POSTFIX

        D_optimizer_kwargs = {
            MODEL_ARG: self.D_optimizer._optimizer if self.use_mixed_precision else self.D_optimizer,
            MODEL_TYPE_ARG: D_model_type,
            **shared_kwargs
        }
        G_optimizer_kwargs = {
            MODEL_ARG: self.G_optimizer._optimizer if self.use_mixed_precision else self.G_optimizer,
            MODEL_TYPE_ARG: G_model_type,
            **shared_kwargs
        }

        save_model(**D_optimizer_kwargs)
        save_model(**G_optimizer_kwargs)

        use_fp16 = self.use_mixed_precision
        if use_fp16:
            OPTIMIZER_ARG = 'optimizer'
            save_optimizer_loss_scale(**{OPTIMIZER_ARG: self.D_optimizer, MODEL_TYPE_ARG: D_model_type, **shared_kwargs})
            save_optimizer_loss_scale(**{OPTIMIZER_ARG: self.G_optimizer, MODEL_TYPE_ARG: G_model_type, **shared_kwargs})

    def save_valid_images(self, models, training_finished_images, smoothed=False):
        G_model, Gs_model = models[G_KEY], models[GS_KEY]

        digits_in_number = 8 # Total number of training images is 25000k for resolution 1024
        fname = ('%0' + str(digits_in_number) + 'd') % training_finished_images

        valid_images_dir = create_images_dir_path(self.model_name + (SMOOTH_POSTFIX if smoothed else ''))
        use_grid_title = False
        if use_grid_title:
            valid_images_grid_title = create_images_grid_title(training_finished_images)
        else:
            valid_images_grid_title = None

        model_kwargs = {'training': False}
        batch_size = 2 * self.batch_size
        if smoothed:
            valid_images = run_model_on_batches(Gs_model, model_kwargs, self.Gs_valid_latents, batch_size)
            if not self.use_gpu_for_Gs:
                valid_images = tf.transpose(valid_images, toNCHW_AXIS)
        else:
            valid_images = run_model_on_batches(G_model, model_kwargs, self.valid_latents, batch_size)
        valid_images = convert_outputs_to_images(
            valid_images, max(2 ** self.resolution_log2, self.min_target_single_image_size),
            hw_ratio=self.dataset_hw_ratio, data_format=self.data_format
        ).numpy()

        save_in_jpg = self.resolution_log2 > self.max_png_res
        fast_save_grid(
            out_dir=valid_images_dir,
            fname=fname,
            images=valid_images,
            title=valid_images_grid_title,
            nrows=self.valid_grid_nrows,
            ncols=self.valid_grid_ncols,
            padding=self.valid_grid_padding,
            save_in_jpg=save_in_jpg
        )

    @tf.function
    def generate_latents(self, batch_size):
        return generate_latents(batch_size, self.latent_size, self.model_compute_dtype)

    @tf.function
    def G_train_step(self, G_model, D_model, evaluate_reg, write_scalars_summary, write_hists_summary, step):
        G_vars = G_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(G_vars, 'Generator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as G_tape:
            G_tape.watch(G_vars)
            G_loss, G_reg = self.G_loss_fn(G_model, D_model, self.G_optimizer,
                                           batch_size=self.batch_size,
                                           evaluate_loss=tf_bool(True),
                                           evaluate_reg=evaluate_reg,
                                           loss_vars=self.loss_vars,
                                           write_summary=write_scalars_summary,
                                           step=step,
                                           **self.G_loss_params)
            if not self.lazy_regularization:
                G_loss += G_reg
            G_loss = maybe_scale_loss(G_loss, self.G_optimizer)

        G_grads = G_tape.gradient(G_loss, G_vars)
        G_grads = maybe_unscale_grads(G_grads, self.G_optimizer)
        self.G_optimizer.apply_gradients(zip(G_grads, G_vars))

        maybe_show_grads_stat(G_grads, G_vars, step, 'G')
        print('Compiled G train step')
        return G_grads

    @tf.function
    def G_reg_train_step(self, G_model, D_model, write_scalars_summary, write_hists_summary, step):
        G_vars = G_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(G_vars, 'Generator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as G_tape:
            G_tape.watch(G_vars)
            _, G_reg = self.G_loss_fn(G_model, D_model, self.G_optimizer,
                                      batch_size=self.batch_size,
                                      evaluate_loss=tf_bool(False),
                                      evaluate_reg=tf_bool(True),
                                      loss_vars=self.loss_vars,
                                      write_summary=write_scalars_summary,
                                      step=step,
                                      **self.G_loss_params)
            G_reg = maybe_scale_loss(self.G_reg_interval * G_reg, self.G_optimizer)

        G_reg_grads = G_tape.gradient(G_reg, G_vars)
        G_reg_grads = maybe_unscale_grads(G_reg_grads, self.G_optimizer)
        self.G_optimizer.apply_gradients(zip(G_reg_grads, G_vars))

        maybe_show_grads_stat(G_reg_grads, G_vars, step, 'G_reg')

        print('Compiled G reg train step')
        return G_reg_grads

    @tf.function
    def D_train_step(self, G_model, D_model, images, evaluate_reg, write_scalars_summary, write_hists_summary, step):
        D_vars = D_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(D_vars, 'Discriminator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as D_tape:
            D_tape.watch(D_vars)
            D_loss, D_reg = self.D_loss_fn(G_model, D_model, self.D_optimizer,
                                           batch_size=self.batch_size,
                                           real_images=images,
                                           evaluate_loss=tf_bool(True),
                                           evaluate_reg=evaluate_reg,
                                           loss_vars=self.loss_vars,
                                           write_summary=write_scalars_summary,
                                           step=step,
                                           **self.D_loss_params)
            if not self.lazy_regularization:
                D_loss += D_reg
            D_loss = maybe_scale_loss(D_loss, self.D_optimizer)

        D_grads = D_tape.gradient(D_loss, D_vars)
        D_grads = maybe_unscale_grads(D_grads, self.D_optimizer)
        self.D_optimizer.apply_gradients(zip(D_grads, D_vars))

        maybe_show_grads_stat(D_grads, D_vars, step, 'D')
        print('Compiled D train step')
        return D_grads

    @tf.function
    def D_reg_train_step(self, G_model, D_model, images, write_scalars_summary, write_hists_summary, step):
        D_vars = D_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(D_vars, 'Discriminator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as D_tape:
            D_tape.watch(D_vars)
            _, D_reg = self.D_loss_fn(G_model, D_model, self.D_optimizer,
                                      batch_size=self.batch_size,
                                      real_images=images,
                                      evaluate_loss=tf_bool(False),
                                      evaluate_reg=tf_bool(True),
                                      loss_vars=self.loss_vars,
                                      write_summary=write_scalars_summary,
                                      step=step,
                                      **self.D_loss_params)

            D_reg = maybe_scale_loss(self.D_reg_interval * D_reg, self.D_optimizer)

        D_reg_grads = D_tape.gradient(D_reg, D_vars)
        D_reg_grads = maybe_unscale_grads(D_reg_grads, self.D_optimizer)
        self.D_optimizer.apply_gradients(zip(D_reg_grads, D_vars))

        maybe_show_grads_stat(D_reg_grads, D_vars, step, 'D_reg')
        print('Compiled D reg train step')
        return D_reg_grads

    def process_hists(self, model, grads, model_name, write_hists_summary, step):
        if write_hists_summary:
            print()
            print('Writing hists summaries...')
            start_time = time.time()

        # Note: it's important to have cond for summaries after name scope definition,
        # otherwise all hists will have the same prefix, e.g. 'cond1'.
        # It holds at least for summaries inside tf.function
        vars = model.trainable_variables
        with tf.device(CPU_DEVICE):
            # Write gradients
            with tf.name_scope(f'{model_name}-grads'):
                if write_hists_summary:
                    for grad, var in zip(grads, vars):
                        hist_grad = tf.cond(is_finite_grad(grad), lambda: grad, lambda: tf.zeros(grad.shape, grad.dtype))
                        tf.summary.histogram(var.name, hist_grad, step=step)
            # Write weights
            with tf.name_scope(f'{model_name}-weights'):
                if write_hists_summary:
                    for var in vars:
                        tf.summary.histogram(var.name, var, step=step)

        if write_hists_summary:
            total_time = time.time() - start_time
            print(f'Hists for {model_name} written in {total_time:.3f} seconds')

    def train_step(self, G_model, D_model, images,
                   evaluate_G_reg, evaluate_D_reg, write_scalars_summary, write_hists_summary, step):
        # Note: explicit use of G and D models allows to make sure that
        # tf.function doesn't compile models (can they be?). Additionally tracing is used
        D_grads = self.D_train_step(G_model, D_model, images,
                                    evaluate_reg=tf_bool(False) if self.lazy_regularization else evaluate_D_reg,
                                    write_scalars_summary=write_scalars_summary,
                                    write_hists_summary=write_hists_summary,
                                    step=step)
        G_grads = self.G_train_step(G_model, D_model,
                                    evaluate_reg=tf_bool(False) if self.lazy_regularization else evaluate_G_reg,
                                    write_scalars_summary=write_scalars_summary,
                                    write_hists_summary=write_hists_summary,
                                    step=step)

        # Note: processing hists summaries inside train_step functions leads to a process crash for large resolutions.
        # Maybe due to compilation. To write hists to TensorBoard they need to be processed separately.
        # This also allows larger batch sizes. Rear OOM warnings don't seem to affect performance
        self.process_hists(D_model, D_grads, 'D', write_hists_summary, step)
        self.process_hists(G_model, G_grads, 'G', write_hists_summary, step)

        # Compute regularization terms in a separate path (only when lazy_regularization is enabled)
        if self.lazy_regularization:
            if evaluate_D_reg:
                D_reg_grads = self.D_reg_train_step(G_model, D_model, images,
                                                    write_scalars_summary=write_scalars_summary,
                                                    write_hists_summary=write_hists_summary,
                                                    step=step)
                self.process_hists(D_model, D_reg_grads, 'D_reg', write_hists_summary, step)

            if evaluate_G_reg:
                G_reg_grads = self.G_reg_train_step(G_model, D_model,
                                                    write_scalars_summary=write_scalars_summary,
                                                    write_hists_summary=write_hists_summary,
                                                    step=step)
                self.process_hists(G_model, G_reg_grads, 'G_reg', write_hists_summary, step)

    def add_resources_summary(self, training_finished_images):
        for device, memory_stats in get_gpu_memory_usage().items():
            for k, v in memory_stats.items():
                if 'peak' in k.lower():
                    self.resources[device] = v
                tf.summary.scalar(f'Resources/{device}/{k}(Mbs)', v, step=training_finished_images)

    def add_timing_summary(self, training_finished_images):
        # 1. Get time info and update last used value
        # One tick is number of images after each scalar summaries is updated
        cur_update_time = time.time()
        tick_time = cur_update_time - self.last_update_time - self.metrics_time
        self.metrics_time = 0.
        self.last_update_time = cur_update_time
        self.cur_tick += 1
        kimg = self.cur_tick * self.summary_scalars_every / 1000.0
        kimg_denom = self.summary_scalars_every / 1000
        total_time = cur_update_time - self.start_time
        # 2. Summary tick time
        # Note: picks on the first call (graph compilation). Metrics time is subtracted
        timing_kimg = tick_time / kimg_denom
        timing_imgs_per_sec = self.summary_scalars_every / tick_time
        tf.summary.scalar(f'Timing/Tick(s)', tick_time, step=training_finished_images)
        tf.summary.scalar(f'Timing/Kimg(s)', timing_kimg, step=training_finished_images)
        tf.summary.scalar(f'Timing/ImgsPerSec', timing_imgs_per_sec, step=training_finished_images)
        # 3. Summary total time
        timing_hours = total_time / (60.0 * 60.0)
        timing_days = total_time / (24.0 * 60.0 * 60.0)
        tf.summary.scalar(f'Timing/Total(hours)', timing_hours, step=training_finished_images)
        tf.summary.scalar(f'Timing/Total(days)', timing_days, step=training_finished_images)
        # 4. Additionally log values
        stats_message = [
            f'tick {self.cur_tick:<5d}',
            f'kimg {kimg:<8.1f}',
            f'time {format_time(total_time):<12s}',
            f'sec/tick {tick_time:<6.1f}',
            f'sec/kimg {timing_kimg:<6.1f}',
            f'imgs/sec {timing_imgs_per_sec:<5.1f}',
        ]
        stats_message += [f'gpumem {k} {int(v):<5d}' for k, v in self.resources.items()]
        if self.use_mixed_precision:
            stats_message += [
                f'G loss scale {int(self.G_optimizer.loss_scale):<4d}',
                f'D loss scale {int(self.D_optimizer.loss_scale):<4d}'
            ]
        print() # Start new line after tqdm pbar
        logging.info(' | '.join(stats_message))

    def post_train_step_actions(self, models, summary_options):
        D_model, G_model, Gs_model = models[D_KEY], models[G_KEY], models[GS_KEY]
        training_finished_images = summary_options[TRAINING_FINISHED_IMAGES_KEY]

        if summary_options[SMOOTH_G_WEIGHTS_KEY]:
            if Gs_model is not None:
                smooth_model_weights(
                    sm_model=Gs_model, src_model=G_model, beta=self.Gs_beta, device=self.Gs_device
                )

        if summary_options[RUN_METRICS_KEY]:
            self.run_metrics(training_finished_images)

        if summary_options[WRITE_LOSS_SCALE_SUMMARY_KEY]:
            tf.summary.scalar('LossScale/D_optimizer', self.D_optimizer.loss_scale, step=training_finished_images)
            tf.summary.scalar('LossScale/G_optimizer', self.G_optimizer.loss_scale, step=training_finished_images)

        if summary_options[WRITE_SCALARS_SUMMARY_KEY]:
            self.add_resources_summary(training_finished_images)
            self.add_timing_summary(training_finished_images)
            self.summary_writer.flush()

        self.maybe_update_loss_scales()
        if summary_options[RESET_LOSS_SCALE_STATS_KEY]:
            self.reset_loss_scale_states()

        if summary_options[SAVE_MODELS_KEY]:
            self.save_models(models=models, step=training_finished_images)
            self.save_optimizers_weights(step=training_finished_images)

        if summary_options[SAVE_VALID_IMAGES_KEY]:
            self.save_valid_images(models, training_finished_images)
            if Gs_model is not None:
                self.save_valid_images(models, training_finished_images, smoothed=True)

    def train_step_options(self, step, stage_steps, n_finished_images):
        first_step_cond          = step == 0
        last_step_cond           = is_last_step(step, stage_steps)
        stage_images             = (step + 1) * self.batch_size
        write_scalars_summary    = should_write_summary(self.summary_scalars_every, stage_images + n_finished_images, self.batch_size) or last_step_cond
        write_loss_scale_summary = self.use_mixed_precision and write_scalars_summary and (not first_step_cond) # The first step usually uses very high scale

        # TODO: should summaries use stage_images or training_finished_images?
        return {
            FIRST_STEP_COND_KEY         : first_step_cond,
            LAST_STEP_COND_KEY          : last_step_cond,
            STAGE_IMAGES_KEY            : stage_images,
            TRAINING_FINISHED_IMAGES_KEY: stage_images + n_finished_images,
            WRITE_LOSS_SCALE_SUMMARY_KEY: write_loss_scale_summary,
            WRITE_SCALARS_SUMMARY_KEY   : write_scalars_summary,
            WRITE_HISTS_SUMMARY_KEY     : should_write_summary(self.summary_hists_every, stage_images + n_finished_images, self.batch_size) or last_step_cond,
            RUN_METRICS_KEY             : should_write_summary(self.run_metrics_every, stage_images + n_finished_images, self.batch_size) or last_step_cond,
            SAVE_MODELS_KEY             : should_write_summary(self.save_model_every, stage_images + n_finished_images, self.batch_size) or last_step_cond,
            SAVE_VALID_IMAGES_KEY       : should_write_summary(self.save_images_every, stage_images + n_finished_images, self.batch_size) or last_step_cond,
            SMOOTH_G_WEIGHTS_KEY        : self.use_Gs,
            RESET_LOSS_SCALE_STATS_KEY  : stage_steps % self.loss_scale_cycle_length == 0, # Cycle length is set in iterations
            EVALUATE_G_REG_KEY          : (self.lazy_regularization and step % self.G_reg_interval == 0) or (not self.lazy_regularization),
            EVALUATE_D_REG_KEY          : (self.lazy_regularization and step % self.D_reg_interval == 0) or (not self.lazy_regularization)
        }

    def init_training_time(self):
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.metrics_time = 0.
        self.cur_tick = -1
        self.resources = {}

    def run_training(self):
        self.init_training_time()
        self.reset_loss_scale_states()

        D_model, G_model, Gs_model = self.create_models()
        n_finished_images          = 0
        training_steps             = int(1000 * self.total_kimages) // self.batch_size
        tf_step                    = tf.Variable(n_finished_images, trainable=False, dtype=tf.int64)

        print('Running training...')

        with self.summary_writer.as_default():
            for step in tqdm(range(training_steps), desc='Training'):
                summary_options = self.train_step_options(step, training_steps, n_finished_images)
                tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                real_images = next(self.images_dataset)
                self.train_step(
                    G_model=G_model, D_model=D_model, images=real_images,
                    evaluate_G_reg=tf_bool(summary_options[EVALUATE_G_REG_KEY]),
                    evaluate_D_reg=tf_bool(summary_options[EVALUATE_D_REG_KEY]),
                    write_scalars_summary=tf_bool(summary_options[WRITE_SCALARS_SUMMARY_KEY]),
                    write_hists_summary=tf_bool(summary_options[WRITE_HISTS_SUMMARY_KEY]),
                    step=tf_step
                )
                self.post_train_step_actions(
                    models={D_KEY: D_model, G_KEY: G_model, GS_KEY: Gs_model}, summary_options=summary_options
                )

        remove_old_models(
            self.model_name, max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        # Save states after extra weights are removed
        self.save_optimizers_weights()

        stabilization_stage_total_time = time.time() - self.start_time
        logging.info(f'Training finished in {format_time(stabilization_stage_total_time)}')

    def run_metrics(self, training_finished_images, summary_writer=None):
        if self.use_Gs:
            G_model = self.Gs_object.create_G_model()
        else:
            G_model = self.G_object.create_G_model()

        if summary_writer is None:
            summary_writer = self.summary_writer
        with summary_writer.as_default():
            metrics_start_time = time.time()
            for idx, metric_object in enumerate(self.metrics_objects):
                metric_name = metric_object.name

                start_time = time.time()
                metric_value = metric_object.run_metric(self.batch_size, G_model)
                total_time = time.time() - start_time

                tf.summary.scalar(f'Metric/{metric_name}', metric_value, step=training_finished_images)
                tf.summary.scalar(f'Metric/{metric_name}/Time(s)', total_time, step=training_finished_images)
                logging.info(f'Evaluated {metric_name} metric in {format_time(total_time)}. Value is {metric_value:.3f} ')

            metrics_total_time = time.time() - metrics_start_time
            tf.summary.scalar(f'Metric/TotalRunTime/Time(s)', metrics_total_time, step=training_finished_images)
            summary_writer.flush()

        self.metrics_time = metrics_total_time

    def run_benchmark_stage(self, images, run_metrics):
        stage_start_time = time.time()
        compile_start_time = time.time()
        compile_total_time = 0
        self.reset_loss_scale_states()

        D_model, G_model, Gs_model = self.create_models()
        benchmark_steps            = images // self.batch_size
        n_finished_images          = 0
        tf_step                    = tf.Variable(0, trainable=False, dtype=tf.int64)

        benchmark_dir = os.path.join('temp_dir', TF_LOGS_DIR)
        summary_writer = tf.summary.create_file_writer(benchmark_dir)
        with summary_writer.as_default():
            for step in tqdm(range(benchmark_steps), desc='Benchmark'):
                # Note: on the 1st step model is compiled, so don't count this time
                if step == 1:
                    stage_start_time = time.time()
                    compile_total_time = stage_start_time - compile_start_time
                    metrics_time = 0.

                summary_options = self.train_step_options(step, benchmark_steps, n_finished_images)
                tf_step.assign(summary_options[TRAINING_FINISHED_IMAGES_KEY])

                real_images = next(self.images_dataset)
                self.train_step(
                    G_model=G_model, D_model=D_model, images=real_images,
                    evaluate_G_reg=tf_bool(summary_options[EVALUATE_G_REG_KEY]),
                    evaluate_D_reg=tf_bool(summary_options[EVALUATE_D_REG_KEY]),
                    write_scalars_summary=tf_bool(summary_options[WRITE_SCALARS_SUMMARY_KEY]),
                    write_hists_summary=tf_bool(summary_options[WRITE_HISTS_SUMMARY_KEY]),
                    step=tf_step
                )

                # Run metrics twice to make sure everything is fine
                if run_metrics and (step == 50 or step == 100):
                    metrics_start_time = time.time()
                    self.run_metrics(-1, summary_writer)
                    metrics_time += time.time() - metrics_start_time

        try:
            shutil.rmtree(benchmark_dir)
        except Exception as e:
            print(e)

        stage_total_time = time.time() - stage_start_time
        train_time = stage_total_time - metrics_time
        print(f'\nBenchmark finished in {format_time(stage_total_time)}. '
              f'Metrics run (2 iterations) in {format_time(metrics_time)}.\n'
              f'Compilation took {format_time(compile_total_time)}. \n'
              f'Training took {format_time(train_time)} for {images} images. In average {(images / train_time):.3f} images/sec.')
