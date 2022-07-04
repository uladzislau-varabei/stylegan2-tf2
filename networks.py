import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from config import Config as cfg
from dnnlib.custom_layers import layer_dtype,\
    fully_connected_layer, conv2d_layer, modulated_conv2d_layer, fused_bias_act_layer, bias_act_layer, const_layer, \
    noise_layer, pixel_norm_layer, downscale2d_layer, upscale2d_layer, minibatch_stddev_layer, resnet_merge_layer, skip_merge_layer
from checkpoint_utils import weights_to_dict, load_model_weights_from_dict
from utils import level_of_details, validate_data_format,\
    get_start_fp16_resolution, should_use_fp16, adjust_clamp,\
    NHWC_FORMAT, NCHW_FORMAT, RESNET_ARCHITECTURE, SKIP_ARCHITECTURE
from tf_utils import generate_latents, get_compute_dtype, lerp, FastTFModel,\
    PER_LAYER_COMPILATION, DEFAULT_DATA_FORMAT, GAIN_INIT_MODE_DICT, GAIN_ACTIVATION_FUNS_DICT, PLOT_MODEL_KWARGS


def n_filters(stage, fmap_base, fmap_decay, fmap_max):
    """
    fmap_base  Overall multiplier for the number of feature maps.
    fmap_decay log2 feature map reduction when doubling the resolution.
    fmap_max   Maximum number of feature maps in any layer.
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class ModelConfig:

    def __init__(self, config):
        self.config = config

        self.target_resolution = config[cfg.TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        # Computations
        self.fused_bias_act             = config.get(cfg.FUSED_BIAS_ACT, cfg.DEFAULT_FUSED_BIAS_ACT)
        self.use_mixed_precision        = config.get(cfg.USE_MIXED_PRECISION, cfg.DEFAULT_USE_MIXED_PRECISION)
        self.num_fp16_resolutions       = config.get(cfg.NUM_FP16_RESOLUTIONS, cfg.DEFAULT_NUM_FP16_RESOLUTIONS)
        self.start_fp16_resolution_log2 =\
            get_start_fp16_resolution(self.num_fp16_resolutions, 2, self.resolution_log2)
        self.model_compute_dtype        = get_compute_dtype(self.use_mixed_precision)
        self.use_xla                    = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
        self.conv_clamp                 = config.get(cfg.CONV_CLAMP, cfg.DEFAULT_CONV_CLAMP)

        self.total_kimages = config[cfg.TOTAL_KIMAGES]
        self.batch_size = config[cfg.BATCH_SIZE]


class GeneratorMapping(ModelConfig):

    def __init__(self, config):
        super(GeneratorMapping, self).__init__(config)

        self.latent_size       = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.dlatent_size      = config.get(cfg.DLATENT_SIZE, cfg.DEFAULT_DLATENT_SIZE)
        self.normalize_latents = config.get(cfg.NORMALIZE_LATENTS, cfg.DEFAULT_NORMALIZE_LATENTS)
        self.mapping_layers    = config.get(cfg.MAPPING_LAYERS, cfg.DEFAULT_MAPPING_LAYERS)
        self.mapping_units     = config.get(cfg.MAPPING_UNITS, cfg.DEFAULT_MAPPING_UNITS)
        self.mapping_lrmul     = config.get(cfg.MAPPING_LRMUL, cfg.DEFAULT_MAPPING_LRMUL)
        self.mapping_act_name  = config.get(cfg.MAPPING_ACTIVATION, cfg.DEFAULT_MAPPING_ACTIVATION)
        self.mapping_gain      = GAIN_ACTIVATION_FUNS_DICT[self.mapping_act_name]
        self.mapping_use_bias  = config.get(cfg.MAPPING_USE_BIAS, cfg.DEFAULT_MAPPING_USE_BIAS)

        self.num_layers = self.resolution_log2 * 2 - 2
        self.num_styles = self.num_layers

    def fc(self, x, units, use_fp16=None, scope=''):
        return fully_connected_layer(x, units, lrmul=self.mapping_lrmul, gain=self.mapping_gain, use_fp16=use_fp16, scope=scope, config=self.config)

    def bias_act(self, x, use_fp16=None, scope=''):
        kwargs = {
            'x': x,
            'act_name': self.mapping_act_name,
            'use_bias': self.mapping_use_bias,
            'lrmul': self.mapping_lrmul,
            'use_fp16': use_fp16,
            'scope': scope,
            'config': self.config
        }
        return fused_bias_act_layer(**kwargs) if self.fused_bias_act else bias_act_layer(**kwargs)

    def create_G_mapping(self):
        # TODO: think about fp16 for this network
        use_fp16 = self.use_mixed_precision

        self.latents = Input([self.latent_size], dtype=self.model_compute_dtype, name='Latents')
        x = self.latents
        if self.normalize_latents:
            #with tf.name_scope('Latents_normalizer') as scope:
            x = pixel_norm_layer(x, use_fp16=use_fp16, config=self.config)

        with tf.name_scope('G_mapping'):
            for layer_idx in range(self.mapping_layers):
                with tf.name_scope(f'FullyConnected{layer_idx}') as scope:
                    units = self.dlatent_size if layer_idx == self.mapping_layers - 1 else self.mapping_units
                    x = self.fc(x, units, use_fp16=use_fp16, scope=scope)
                    x = self.bias_act(x, use_fp16=use_fp16, scope=scope)

        with tf.name_scope('Broadcast'):
            x = tf.tile(x[:, tf.newaxis], [1, self.num_styles, 1])

        dlatents = tf.identity(x, name='dlatents')
        if PER_LAYER_COMPILATION:
            self.G_mapping = tf.keras.Model(self.latents, dlatents, name='G_mapping')
        else:
            self.G_mapping = FastTFModel(self.latents, dlatents, use_xla=self.use_xla, name='G_mapping')


class GeneratorStyle(tf.keras.Model, ModelConfig):

    def __init__(self, G_mapping, G_synthesis, config):
        ModelConfig.__init__(self, config)
        tf.keras.Model.__init__(self, name='G_style')

        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis

        self.latent_size         = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.dlatent_size        = config.get(cfg.DLATENT_SIZE, cfg.DEFAULT_DLATENT_SIZE)
        self.randomize_noise     = config.get(cfg.RANDOMIZE_NOISE, cfg.DEFAULT_RANDOMIZE_NOISE)
        self.truncation_psi      = config.get(cfg.TRUNCATION_PSI, cfg.DEFAULT_TRUNCATION_PSI)
        self.truncation_cutoff   = config.get(cfg.TRUNCATION_CUTOFF, cfg.DEFAULT_TRUNCATION_CUTOFF)
        self.dlatent_avg_beta    = config.get(cfg.DLATENT_AVG_BETA, cfg.DEFAULT_DLATENT_AVG_BETA)
        self.style_mixing_prob   = config.get(cfg.STYLE_MIXING_PROB, cfg.DEFAULT_STYLE_MIXING_PROB)

        self.num_layers = self.resolution_log2 * 2 - 2
        self.res_num_layers = self.num_layers

        self.validate_call_params()
        self.initialize_variables()

    def validate_call_params(self):
        def validate_range(value, min_val, max_val):
            if value is not None:
                assert min_val <= value <= max_val
        validate_range(self.truncation_psi, 0.0, 1.0)
        # validate_range(self.truncation_cutoff, 0, self.num_layers)
        validate_range(self.dlatent_avg_beta, 0.0, 1.0)
        validate_range(self.style_mixing_prob, 0.0, 1.0)

    def initialize_variables(self):
        self.dlatent_avg = self.add_weight(
            name='dlatent_avg',
            shape=[self.dlatent_size],
            dtype=self.model_compute_dtype,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        # Prepare value for style mixing and truncation
        self.layer_idx = tf.range(self.num_layers)[tf.newaxis, :, tf.newaxis]

        if self.style_mixing_prob is not None:
            self.mixing_cur_layers = self.res_num_layers

        if (self.truncation_psi is not None) and (self.truncation_cutoff is not None):
            ones = tf.ones(self.layer_idx.shape, dtype=self.model_compute_dtype)
            self.truncation_coefs = tf.where(self.layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)

    def update_dlatent_avg(self, dlatents):
        batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
        self.dlatent_avg.assign(
            lerp(batch_avg, self.dlatent_avg, self.dlatent_avg_beta)
        )

    def generate_latents(self, batch_size):
        return generate_latents(batch_size, self.latent_size, self.model_compute_dtype)

    def apply_style_mixing(self, dlatents):
        latents2 = self.generate_latents(tf.shape(dlatents)[0])
        # Styles can only be mixed during training
        dlatents2 = self.G_mapping(latents2, training=True)
        mixing_cutoff = tf.cond(
            tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob,
            lambda: tf.random.uniform([], 1, self.mixing_cur_layers, dtype=tf.int32),
            lambda: self.mixing_cur_layers
        )
        dlatents = tf.where(tf.broadcast_to(self.layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
        return dlatents

    def apply_truncation_trick(self, dlatents):
        return lerp(self.dlatent_avg, dlatents, self.truncation_coefs)

    def call(self, latents, return_dlatents=False, training=True, validation=False, truncation_psi=None, truncation_cutoff=None, *args, **kwargs):
        # 1. Decide which actions to perform based on training/testing/validation.
        # Validation is used for metrics evaluation. Testing for generation of images
        assert not (training and validation), "Model can't use training and validation modes at the same time"
        if training or validation:
            truncation_psi = None
            truncation_cutoff = None
        else:
            truncation_psi = self.truncation_psi if truncation_psi is None else truncation_psi
            truncation_cutoff = self.truncation_cutoff if truncation_cutoff is None else truncation_cutoff

        should_update_dlatent_avg = (self.dlatent_avg_beta is not None) and training
        should_apply_style_mixing = (self.style_mixing_prob is not None) and training
        should_apply_truncation_trick = (truncation_psi is not None) and (truncation_cutoff is not None)

        # 2. Evaluate dlatents, output shape: (batch, num_layers, dlatent_size)
        dlatents = self.G_mapping(latents, training=training)

        # 3. Update moving average of W
        with tf.name_scope('DlatentAvg'):
            if should_update_dlatent_avg:
                self.update_dlatent_avg(dlatents)

        # 4. Perform mixing style regularization
        with tf.name_scope('StyleMixing'):
            if should_apply_style_mixing:
                dlatents = self.apply_style_mixing(dlatents)

        # 5. Apply truncation trick
        with tf.name_scope('Truncation'):
            if should_apply_truncation_trick:
                dlatents = self.apply_truncation_trick(dlatents)

        # 6. Evaluate synthesis network
        images_out = self.G_synthesis(dlatents, training=training)

        if return_dlatents:
            return images_out, dlatents
        return images_out


class Generator(ModelConfig):

    def __init__(self, config):
        super(Generator, self).__init__(config)

        self.G_architecture    = config.get(cfg.G_ARCHITECTURE, cfg.DEFAULT_G_ARCHITECTURE)
        assert self.G_architecture in [SKIP_ARCHITECTURE, RESNET_ARCHITECTURE], \
            f"Architecture {self.G_architecture} is not supported"
        self.latent_size       = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
        self.dlatent_size      = config.get(cfg.DLATENT_SIZE, cfg.DEFAULT_DLATENT_SIZE)
        self.fmap_const        = config.get(cfg.FMAP_CONST, cfg.DEFAULT_FMAP_CONST)
        self.normalize_latents = config.get(cfg.NORMALIZE_LATENTS, cfg.DEFAULT_NORMALIZE_LATENTS)
        self.use_noise         = config.get(cfg.USE_NOISE, cfg.DEFAULT_USE_NOISE)
        self.randomize_noise   = config.get(cfg.RANDOMIZE_NOISE, cfg.DEFAULT_RANDOMIZE_NOISE)
        self.use_bias          = config.get(cfg.USE_BIAS, cfg.DEFAULT_USE_BIAS)
        self.G_kernel_size     = config.get(cfg.G_KERNEL_SIZE, cfg.DEFAULT_G_KERNEL_SIZE)
        self.G_fmap_base       = config.get(cfg.G_FMAP_BASE, cfg.DEFAULT_FMAP_BASE)
        self.G_fmap_decay      = config.get(cfg.G_FMAP_DECAY, cfg.DEFAULT_FMAP_DECAY)
        self.G_fmap_max        = config.get(cfg.G_FMAP_MAX, cfg.DEFAULT_FMAP_MAX)
        self.G_act_name        = config.get(cfg.G_ACTIVATION, cfg.DEFAULT_G_ACTIVATION)

        self.G_weights_init_mode = config.get(cfg.G_WEIGHTS_INIT_MODE, None)
        if self.G_weights_init_mode is None:
            self.G_gain = GAIN_ACTIVATION_FUNS_DICT[self.G_act_name]
        else:
            self.G_gain = GAIN_INIT_MODE_DICT[self.G_weights_init_mode]

        self.num_layers = self.resolution_log2 * 2 - 2
        self.G_model = None
        self.create_model_layers()
        self.create_G_model()

    def G_n_filters(self, stage):
        return n_filters(stage, self.G_fmap_base, self.G_fmap_decay, self.G_fmap_max)

    def G_output_shape(self, res):
        nf = self.G_n_filters(res - 1)
        if self.data_format == NCHW_FORMAT:
            return [nf, 2 ** res, 2 ** res]
        else: # self.data_format == NHWC_FORMAT:
            return [2 ** res, 2 ** res, nf]

    def should_use_fp16(self, res):
        return should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)

    def create_model_layers(self):
        self.G_mapping_object = GeneratorMapping(self.config)
        self.G_mapping_object.create_G_mapping()
        self.latents = self.G_mapping_object.latents
        self.G_mapping = self.G_mapping_object.G_mapping
        # Use mapping network to get shape of dlatents
        self.dlatents = Input(self.G_mapping.output_shape[1:], dtype=self.model_compute_dtype, name='Dlatents')
        self.toRGB_layers = {res: self.to_rgb_layer(res) for res in range(2, self.resolution_log2 + 1)}

    def to_rgb_layer(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'ToRGB_lod{lod}'
        use_fp16 = self.should_use_fp16(res)

        with tf.name_scope(block_name) as scope:
            x = Input(self.G_output_shape(res))
            y = self.modulated_conv2d(x, self.dlatents[:, res * 2 - 3], fmaps=3, kernel_size=1, demodulate=False, gain=1., use_fp16=use_fp16, scope=scope)
            y = self.bias_act(y, act_name='linear', clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)

        return tf.keras.Model([x, self.dlatents], y, name=block_name)

    def to_rgb(self, x, y, res):
        t = self.toRGB_layers[res]([x, self.dlatents])
        if y is not None:
            use_fp16 = self.should_use_fp16(res)
            t = skip_merge_layer(t, y, use_fp16=use_fp16, name=f'{2**res}x{2**res}/Skip_merge/Lambda') # Looks better in TensorBoard
        return t

    def fc(self, x, units, gain=None, use_fp16=None, scope=''):
        if gain is None: gain = self.G_gain
        return fully_connected_layer(x, units, gain=gain, use_fp16=use_fp16, scope=scope, config=self.config)

    def conv2d(self, x, fmaps, kernel_size=None, gain=None, up=False, use_fp16=None, scope=''):
        if kernel_size is None: kernel_size = self.G_kernel_size
        if gain is None: gain = self.G_gain
        return conv2d_layer(
            x, fmaps, kernel_size=kernel_size, gain=gain, up=up,
            use_fp16=use_fp16, scope=scope, config=self.config
        )

    def modulated_conv2d(self, x, dlatents, fmaps, kernel_size=None, up=False, demodulate=True, gain=None, use_fp16=None, scope=''):
        if kernel_size is None: kernel_size = self.G_kernel_size
        if gain is None: gain = self.G_gain
        return modulated_conv2d_layer(
            x, dlatents, fmaps=fmaps, kernel_size=kernel_size, up=up, demodulate=demodulate,
            use_fp16=use_fp16, gain=gain, scope=scope, config=self.config
        )

    def upscale2d(self, x, use_fp16=None, name=None):
        return upscale2d_layer(x, factor=2, use_fp16=use_fp16, config=self.config, name=name)

    def bias_act(self, x, act_name=None, clamp=None, use_fp16=None, scope=''):
        if act_name is None: act_name = self.G_act_name
        clamp = adjust_clamp(clamp, use_fp16)
        kwargs = {
            'x': x,
            'act_name': act_name,
            'use_bias': self.use_bias,
            'clamp': clamp,
            'use_fp16': use_fp16,
            'scope': scope,
            'config': self.config
        }
        return fused_bias_act_layer(**kwargs) if self.fused_bias_act else bias_act_layer(**kwargs)

    def style_block(self, x, layer_idx, fmaps, up=False, use_fp16=None, scope=''):
        x = self.modulated_conv2d(x, self.dlatents[:, layer_idx], fmaps=fmaps, up=up, use_fp16=use_fp16, scope=scope)
        if self.use_noise:
            x = noise_layer(x, use_fp16=use_fp16, scope=scope, config=self.config)
        x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
        return x

    def input_G_block(self):
        use_fp16 = self.should_use_fp16(2)
        y = None
        with tf.name_scope('4x4'):
            with tf.name_scope('Const') as scope:
                fmaps = self.fmap_const if self.fmap_const is not None else self.G_n_filters(1)
                x = const_layer(self.dlatents[:, 0], fmaps, use_fp16=use_fp16, scope=scope, config=self.config)
            with tf.name_scope('Conv') as scope:
                x = self.style_block(x, layer_idx=0, fmaps=self.G_n_filters(1), use_fp16=use_fp16, scope=scope)
            if self.G_architecture == SKIP_ARCHITECTURE:
                y = self.to_rgb(x, y, 2)
        return x, y

    def G_block(self, x, res):
        # res = 3 ... resolution_log2
        t = x
        use_fp16 = self.should_use_fp16(res)
        with tf.name_scope(f'{2**res}x{2**res}') as top_scope:
            with tf.name_scope('Conv0_up') as scope:
                x = self.style_block(x, res * 2 - 5, fmaps=self.G_n_filters(res - 1), up=True, use_fp16=use_fp16, scope=scope)
            with tf.name_scope('Conv1') as scope:
                x = self.style_block(x, res * 2 - 4, fmaps=self.G_n_filters(res - 1), use_fp16=use_fp16, scope=scope)
            if self.G_architecture == RESNET_ARCHITECTURE:
                with tf.name_scope('Skip') as scope:
                    t = self.conv2d(t, fmaps=self.G_n_filters(res - 1), kernel_size=1, up=True, use_fp16=use_fp16, scope=scope)
                    x = resnet_merge_layer(x, t, use_fp16=use_fp16, name=f'{2**res}x{2**res}/Resnet_merge/Lambda') # Looks better in TensorBoard
        return x

    def create_G_model(self):
        if self.G_model is None:
            print(f' ...Creating G model...')
            x, y = self.input_G_block()
            for res in range(3, self.resolution_log2 + 1):
                use_fp16 = self.should_use_fp16(res)
                x = self.G_block(x, res)
                if self.G_architecture == SKIP_ARCHITECTURE:
                    y = self.upscale2d(y, use_fp16, name=f'Upscale2d_{2**res}x{2**res}')
                if self.G_architecture == SKIP_ARCHITECTURE or res == self.resolution_log2:
                    y = self.to_rgb(x, y, res)
            images_out = tf.identity(y, name='images_out')
            # Build models
            if PER_LAYER_COMPILATION:
                self.G_synthesis = tf.keras.Model(self.dlatents, images_out, name='G_synthesis')
            else:
                self.G_synthesis = FastTFModel(self.dlatents, images_out, use_xla=self.use_xla, name='G_synthesis')
            self.G_model = GeneratorStyle(self.G_mapping, self.G_synthesis, self.config)
        return self.G_model

    def initialize_G_model(self, plot_model=True):
        latents = tf.zeros(shape=[self.batch_size, self.latent_size], dtype=self.model_compute_dtype)
        _ = self.G_model(latents, training=False)
        if plot_model:
            try:
                tf.keras.utils.plot_model(self.G_synthesis, to_file='G_synthesis.png', **PLOT_MODEL_KWARGS)
                tf.keras.utils.plot_model(self.G_model, to_file='G_model.png', **PLOT_MODEL_KWARGS)
            except Exception as e:
                print('No images with G models architecture')
                print(e)
        print(f' ...Initialized G model...')

    def save_G_weights_in_class(self, G_model):
        self.G_weights_dict = weights_to_dict(G_model)

    def load_G_weights_from_class(self, G_model):
        return load_model_weights_from_dict(G_model, self.G_weights_dict)

    def trace_G_graph(self, summary_writer, writer_dir):
        trace_G_input = tf.zeros(shape=[1, self.dlatent_size], dtype=self.model_compute_dtype)
        trace_G_model = tf.function(self.G_model)
        print('Tracing Generator model...')
        with summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)
            trace_G_model(trace_G_input)
            tf.summary.trace_export(
                name='Generator',
                step=0,
                profiler_outdir=writer_dir
            )
            tf.summary.trace_off()
            summary_writer.flush()
        print('Generator model traced!')


class Discriminator(ModelConfig):

    def __init__(self, config):
        super(Discriminator, self).__init__(config)

        self.D_architecture   = config.get(cfg.D_ARCHITECTURE, cfg.DEFAULT_D_ARCHITECTURE)
        assert self.D_architecture in [SKIP_ARCHITECTURE, RESNET_ARCHITECTURE], \
            f"Architecture {self.D_architecture} is not supported"
        self.use_bias         = config.get(cfg.USE_BIAS, cfg.DEFAULT_USE_BIAS)
        self.mbstd_group_size = config[cfg.MBSTD_GROUP_SIZE]
        self.D_kernel_size    = config.get(cfg.D_KERNEL_SIZE, cfg.DEFAULT_D_KERNEL_SIZE)
        self.D_fmap_base      = config.get(cfg.D_FMAP_BASE, cfg.DEFAULT_FMAP_BASE)
        self.D_fmap_decay     = config.get(cfg.D_FMAP_DECAY, cfg.DEFAULT_FMAP_DECAY)
        self.D_fmap_max       = config.get(cfg.D_FMAP_MAX, cfg.DEFAULT_FMAP_MAX)
        self.D_act_name       = config.get(cfg.D_ACTIVATION, cfg.DEFAULT_D_ACTIVATION)

        self.D_weights_init_mode = config.get(cfg.D_WEIGHTS_INIT_MODE, None)
        if self.D_weights_init_mode is None:
            self.D_gain = GAIN_ACTIVATION_FUNS_DICT[self.D_act_name]
        else:
            self.D_gain = GAIN_INIT_MODE_DICT[self.D_weights_init_mode]

        self.D_model = None
        self.create_model_layers()
        self.create_D_model()

    def D_n_filters(self, stage):
        return n_filters(stage, self.D_fmap_base, self.D_fmap_decay, self.D_fmap_max)

    def D_input_shape(self, res):
        if self.data_format == NCHW_FORMAT:
            return [3, 2 ** res, 2 ** res]
        else: # self.data_format == NHWC_FORMAT:
            return [2 ** res, 2 ** res, 3]

    def should_use_fp16(self, res):
        return should_use_fp16(res, self.start_fp16_resolution_log2, self.use_mixed_precision)

    def create_model_layers(self):
        self.D_input_layer = Input(shape=self.D_input_shape(self.resolution_log2), dtype=self.model_compute_dtype,
                                   name=f'Images_{2**self.resolution_log2}x{2**self.resolution_log2}')
        self.fromRGB_layers = {res: self.from_rgb_layer(res) for res in range(2, self.resolution_log2 + 1)}

    def from_rgb_layer(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'FromRGB_lod{lod}'
        use_fp16 = self.should_use_fp16(res)

        with tf.name_scope(block_name) as scope:
            x = Input(self.D_input_shape(res))
            y = self.conv2d(x, fmaps=self.D_n_filters(res - 1), kernel_size=1, use_fp16=use_fp16, scope=scope)
            y = self.bias_act(y, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)

        return tf.keras.Model(x, y, name=block_name)

    def from_rgb(self, x, y, res):
        t = self.fromRGB_layers[res](y)
        if x is not None:
            use_fp16 = self.should_use_fp16(res)
            t = skip_merge_layer(x, t, use_fp16=use_fp16, name=f'{2**res}x{2**res}/Skip_merge/Lambda') # Looks better in TensorBoard
        return t

    def fc(self, x, units, gain=None, use_fp16=None, scope=''):
        if gain is None: gain = self.D_gain
        return fully_connected_layer(x, units, gain=gain, use_fp16=use_fp16, scope=scope, config=self.config)

    def conv2d(self, x, fmaps, kernel_size=None, gain=None, down=False, use_fp16=None, scope=''):
        if kernel_size is None: kernel_size = self.D_kernel_size
        if gain is None: gain = self.D_gain
        return conv2d_layer(
            x, fmaps, kernel_size=kernel_size, gain=gain, down=down,
            use_fp16=use_fp16, scope=scope, config=self.config
        )

    def downscale2d(self, x, use_fp16=None, name=None):
        return downscale2d_layer(x, factor=2, use_fp16=use_fp16, config=self.config, name=name)

    def bias_act(self, x, act_name=None, clamp=None, use_fp16=None, scope=''):
        if act_name is None: act_name = self.D_act_name
        clamp = adjust_clamp(clamp, use_fp16)
        kwargs = {
            'x': x,
            'act_name': act_name,
            'use_bias': self.use_bias,
            'clamp': clamp,
            'use_fp16': use_fp16,
            'scope': scope,
            'config': self.config
        }
        return fused_bias_act_layer(**kwargs) if self.fused_bias_act else bias_act_layer(**kwargs)

    def D_block(self, x, res):
        # 8x8 and up
        t = x
        use_fp16 = self.should_use_fp16(res)
        with tf.name_scope(f'{2**res}x{2**res}') as top_scope:
            with tf.name_scope('Conv') as scope:
                x = self.conv2d(x, fmaps=self.D_n_filters(res - 1), use_fp16=use_fp16, scope=scope)
                x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
            with tf.name_scope('Conv1_down') as scope:
                x = self.conv2d(x, fmaps=self.D_n_filters(res - 2), down=True, use_fp16=use_fp16, scope=scope)
                x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
            if self.D_architecture == RESNET_ARCHITECTURE:
                with tf.name_scope('Skip') as scope:
                    t = self.conv2d(t, fmaps=self.D_n_filters(res - 2), kernel_size=1, down=True, use_fp16=use_fp16, scope=scope)
                    x = resnet_merge_layer(x, t, use_fp16=use_fp16, name=f'{2**res}x{2**res}/Resnet_merge/Lambda') # Looks better in TensorBoard
        return x

    def output_D_block(self, x, y):
        # 4x4
        use_fp16 = self.should_use_fp16(2)
        with tf.name_scope('4x4') as top_scope:
            if self.D_architecture == SKIP_ARCHITECTURE:
                x = self.from_rgb(x, y, 2)
            if self.mbstd_group_size > 1:
                x = minibatch_stddev_layer(x, use_fp16, scope=top_scope, config=self.config)
            with tf.name_scope('Conv') as scope:
                x = self.conv2d(x, fmaps=self.D_n_filters(1), use_fp16=use_fp16, scope=scope)
                x = self.bias_act(x, clamp=self.conv_clamp, use_fp16=use_fp16, scope=scope)
            with tf.name_scope('FullyConnected0') as scope:
                x = self.fc(x, units=self.D_n_filters(0), use_fp16=use_fp16, scope=scope)
                x = self.bias_act(x, use_fp16=use_fp16, scope=scope)
            with tf.name_scope('FullyConnected1') as scope:
                x = self.fc(x, units=1, gain=1., use_fp16=use_fp16, scope=scope)
                x = self.bias_act(x, act_name='linear', use_fp16=use_fp16, scope=scope)
        return x

    def create_D_model(self):
        if self.D_model is None:
            print(f' ...Creating D model...')
            inputs = self.D_input_layer
            x = None
            y = inputs
            # Build for the main resolutions.
            for res in range(self.resolution_log2, 2, -1):
                use_fp16 = self.should_use_fp16(res)
                if self.D_architecture == SKIP_ARCHITECTURE or res == self.resolution_log2:
                    x = self.from_rgb(x, y, res)
                x = self.D_block(x, res)
                if self.D_architecture == SKIP_ARCHITECTURE:
                    y = self.downscale2d(y, use_fp16=use_fp16, name=f'Downscale2d_{2**res}x{2**res}')
            # Build for the final layers.
            x = self.output_D_block(x, y)
            # Build final model.
            scores_out = tf.identity(x, name='scores_out')
            if PER_LAYER_COMPILATION:
                self.D_model = tf.keras.Model(inputs, scores_out, name='D_style')
            else:
                self.D_model = FastTFModel(inputs, scores_out, use_xla=self.use_xla, name='D_style')
        return self.D_model

    def initialize_D_model(self):
        images = tf.zeros(shape=[self.batch_size] + self.D_input_shape(self.resolution_log2), dtype=self.model_compute_dtype)
        _ = self.D_model(images, training=False)
        try:
            tf.keras.utils.plot_model(self.D_model, to_file='D_model.png', **PLOT_MODEL_KWARGS)
        except Exception as e:
            print('No image with D model architecture')
            print(e)
        print(f' ...Initialized D model... ')

    def save_D_weights_in_class(self, D_model):
        self.D_weights_dict = weights_to_dict(D_model)

    def load_D_weights_from_class(self, D_model):
        return load_model_weights_from_dict(D_model, self.D_weights_dict)

    def trace_D_graph(self, summary_writer, writer_dir):
        trace_D_input = tf.zeros([1] + self.D_input_shape(self.resolution_log2), dtype=self.model_compute_dtype)
        trace_D_model = tf.function(self.D_model)
        print('Tracing Discriminator model...')
        with summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)
            trace_D_model(trace_D_input)
            tf.summary.trace_export(
                name='Discriminator',
                step=0,
                profiler_outdir=writer_dir
            )
            tf.summary.trace_off()
            summary_writer.flush()
        print('Discriminator model traced!')
