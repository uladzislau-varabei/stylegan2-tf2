import os
import logging
import platform

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from utils import NCHW_FORMAT, NHWC_FORMAT,\
    should_log_debug_info, to_hw_size, validate_data_format, validate_hw_ratio


PER_LAYER_COMPILATION = False
DEFAULT_DATA_FORMAT = NCHW_FORMAT
MAX_LOSS_SCALE = 2 ** 15 # Max loss scale taken from source code for LossScaleOptimizer. Valid for TF 2.5

# NCHW -> NHWC
toNHWC_AXIS = [0, 2, 3, 1]
# NHWC -> NCHW
toNCHW_AXIS = [0, 3, 1, 2]

OS_LINUX = 'Linux'
OS_WIN = 'Windows'
RANDOM_NOISE_WEIGHT = 'random_noise_weight'

PLOT_MODEL_KWARGS = {
    'show_shapes': True,
    'show_dtype': True,
    'show_layer_names': True,
    'rankdir': 'TB',
    'expand_nested': True,
    'dpi': 96
}


#----------------------------------------------------------------------------
# Activation functions utils.

HE_INIT    = 'He'
LECUN_INIT = 'LeCun'

HE_GAIN    = np.sqrt(2.)
LECUN_GAIN = 1.

GAIN_INIT_MODE_DICT = {
    LECUN_INIT: LECUN_GAIN,
    HE_INIT   : HE_GAIN
}

GAIN_ACTIVATION_FUNS_DICT = {
    'relu'      : HE_GAIN,
    # The same 2 functions
    'leaky_relu': HE_GAIN,
    'lrelu'     : HE_GAIN,
    # The same 2 functions
    'swish'     : HE_GAIN,
    'silu'      : HE_GAIN,
    'gelu'      : HE_GAIN,
    'mish'      : HE_GAIN,
    # The same 2 functions
    'hard_mish' : HE_GAIN,
    'hmish'     : HE_GAIN,
    # The same 2 functions
    'hard_swish': HE_GAIN,
    'hswish'    : HE_GAIN,
    # A special gain is to be used by default
    'selu'      : LECUN_GAIN
}


@tf.custom_gradient
def mish(x):
    x = tf.convert_to_tensor(x, name="features")

    # Inspired by source code for tf.nn.swish
    def grad(dy):
        """ Gradient for the Mish activation function"""
        # Naively, x * tf.nn.tanh(tf.nn.softplus(x)) requires keeping both
        # x and tf.nn.tanh(tf.nn.softplus(x)) around for backprop,
        # effectively doubling the tensor's memory consumption.
        # We use a control dependency here so that tanh(softplus(x)) is re-computed
        # during backprop (the control dep prevents it being de-duped with the
        # forward pass) and we can free the tanh(softplus(x)) expression immediately
        # after use during the forward pass.
        with tf.control_dependencies([dy]):
            x_tanh_sp = tf.nn.tanh(tf.nn.softplus(x))
        x_sigmoid = tf.nn.sigmoid(x)
        activation_grad = x_tanh_sp + x * x_sigmoid * (1.0 - x_tanh_sp * x_tanh_sp)
        return dy * activation_grad

    return x * tf.nn.tanh(tf.nn.softplus(x)), grad


def hard_mish(x):
    x = tf.convert_to_tensor(x, name="features")
    return tf.minimum(2.0, tf.nn.relu(x + 2.0)) * 0.5 * x


def hard_swish(x):
    x = tf.convert_to_tensor(x, name="features")
    return x * tf.nn.relu6(x + 3.0) * 0.16666667


FUNC_KEY = 'func'
GAIN_KEY = 'gain'
ACTIVATION_FUNCS_DICT = {
    'linear'    : {FUNC_KEY: lambda x: x,                                GAIN_KEY: 1.0},
    'relu'      : {FUNC_KEY: lambda x: tf.nn.relu(x),                    GAIN_KEY: np.sqrt(2.0)},
    # The same 2 functions
    'leaky_relu': {FUNC_KEY: lambda x: tf.nn.leaky_relu(x, alpha=0.2),   GAIN_KEY: np.sqrt(2.0)},
    'lrelu'     : {FUNC_KEY: lambda x: tf.nn.leaky_relu(x, alpha=0.2),   GAIN_KEY: np.sqrt(2.0)},
    'selu'      : {FUNC_KEY: lambda x: tf.nn.selu(x),                    GAIN_KEY: 1.0},
    # The same 2 functions
    'swish'     : {FUNC_KEY: lambda x: tf.nn.swish(x),                   GAIN_KEY: np.sqrt(2.0)},
    'silu'      : {FUNC_KEY: lambda x: tf.nn.swish(x),                   GAIN_KEY: np.sqrt(2.0)},
    'gelu'      : {FUNC_KEY: lambda x: tf.nn.gelu(x, approximate=False), GAIN_KEY: np.sqrt(2.0)},
    'mish'      : {FUNC_KEY: lambda x: mish(x),                          GAIN_KEY: np.sqrt(2.0)},
    # The same 2 functions
    'hard_mish' : {FUNC_KEY: lambda x: hard_mish(x),                     GAIN_KEY: np.sqrt(2.0)},
    'hmish'     : {FUNC_KEY: lambda x: hard_mish(x),                     GAIN_KEY: np.sqrt(2.0)},
    # The same 2 functions
    'hard_swish': {FUNC_KEY: lambda x: hard_swish(x),                    GAIN_KEY: np.sqrt(2.0)},
    'hswish'    : {FUNC_KEY: lambda x: hard_swish(x),                    GAIN_KEY: np.sqrt(2.0)}
}

# Activation function which (might?) need to use float32 dtype
# Should this activation only use fp32?
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
FP32_ACTIVATIONS = ['selu']


# See https://www.tensorflow.org/probability/api_docs/python/tfp/math/clip_by_value_preserve_gradient
def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max, name=None):
    with tf.name_scope(name or 'clip_by_value_preserve_gradient'):
        return t + tf.stop_gradient(tf.clip_by_value(t, clip_value_min, clip_value_max) - t)


def naive_upsample(x, factor, data_format=DEFAULT_DATA_FORMAT):
    assert isinstance(factor, int) and factor >= 1
    validate_data_format(data_format)

    # No op => early exit
    if factor == 1:
        return x

    if data_format == NCHW_FORMAT:
        _, c, h, w = x.shape
        pretile_shape = [-1, c, h, 1, w, 1]
        tile_mults = [1, 1, 1, factor, 1, factor]
        output_shape = [-1, c, h * factor, w * factor]
    else:  # data_format == NHWC_FORMAT:
        _, h, w, c = x.shape
        pretile_shape = [-1, h, 1, w, 1, c]
        tile_mults = [1, 1, factor, 1, factor, 1]
        output_shape = [-1, h * factor, w * factor, c]

    x = tf.reshape(x, pretile_shape)
    x = tf.tile(x, tile_mults)
    x = tf.reshape(x, output_shape)
    return x


def naive_downsample(x, factor, data_format=DEFAULT_DATA_FORMAT):
    assert isinstance(factor, int) and factor >= 1
    validate_data_format(data_format)

    s = tf.shape(x)
    if data_format == NCHW_FORMAT:
        c, h, w = s[1], s[2], s[3]
        target_shape = [-1, c, h // factor, factor, w // factor, factor]
        axis = [3, 5]
    else: # data_format == NHWC_FORMAT:
        h, w, c = s[1], s[2], s[3]
        target_shape = [-1, h // factor, factor, w // factor, factor, c]
        axis = [2, 4]

    x = tf.reshape(x, target_shape)
    x = tf.reduce_mean(x, axis=axis)
    return x


#----------------------------------------------------------------------------
# Model utils.

def enable_random_noise(model: tf.keras.Model):
    for var in model.variables:
        if RANDOM_NOISE_WEIGHT in var.name:
            var.assign(1.)


def disable_random_noise(model: tf.keras.Model):
    for var in model.variables:
        if RANDOM_NOISE_WEIGHT in var.name:
            var.assign(0.)



@tf.function
def smooth_model_weights(sm_model, src_model, beta, device):
    trace_message(' ...Tracing smoothing weights... ')
    smoothed_net_vars = sm_model.trainable_variables
    source_net_vars = src_model.trainable_variables
    trace_vars(smoothed_net_vars, 'Smoothed vars:')
    trace_vars(source_net_vars, 'Source vars:')
    with tf.device(device):
        for a, b in zip(smoothed_net_vars, source_net_vars):
            a.assign(lerp(b, a, beta))


class FastTFModel(tf.keras.Model):

    def __init__(self, x, y, use_xla, name):
        super(FastTFModel, self).__init__(name='Wrapped_' + name)
        self.model = tf.keras.Model(x, y, name=name)
        self.use_xla = use_xla
        self.call = tf.function(self.call, jit_compile=use_xla)

    @property
    def input_shape(self):
        return self.model.input_shape

    @property
    def output_shape(self):
        return self.model.output_shape

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    def call(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)


#----------------------------------------------------------------------------
# Grads / Optimizer utils.

def set_optimizer_iters(optimizer, iters):
    weights = optimizer.get_weights()
    if len(weights) > 0:
        weights[0] = iters
        optimizer.set_weights(weights)


def maybe_scale_loss(loss, optimizer):
    return optimizer.get_scaled_loss(loss) if optimizer.use_mixed_precision else loss


def maybe_unscale_grads(grads, optimizer):
    return optimizer.get_unscaled_gradients(grads) if optimizer.use_mixed_precision else grads


def custom_unscale_grads(grads, vars, optimizer: mixed_precision.LossScaleOptimizer):
    # Note: works inside tf.function
    # All grads are casted to fp32
    dtype = tf.float32
    coef = fp32(1. / optimizer.loss_scale)

    def unscale_grad(g, v):
        return fp32(g) * coef if g is not None else (tf.zeros_like(v, dtype=dtype), v)

    grads_array = tf.TensorArray(dtype, len(grads))
    for i in tf.range(len(grads)):
        grads_array = grads_array.write(i, unscale_grad(grads[i], vars[i]))

    return grads_array.stack()


def maybe_custom_unscale_grads(grads, vars, optimizer):
    return custom_unscale_grads(grads, vars, optimizer) if optimizer.use_mixed_precision else grads


def is_finite_grad(grad):
    return tf.equal(tf.math.count_nonzero(~tf.math.is_finite(grad)), 0)


def should_update_loss_scale(optimizer, threshold_scale):
    state = True
    if optimizer.use_mixed_precision:
        if optimizer.dynamic:  # Bool indicating whether dynamic loss scaling is used.
            return optimizer.loss_scale < threshold_scale
    return state


def update_loss_scale(optimizer, name):
    try:
        if optimizer.use_mixed_precision:
            if optimizer.dynamic: # Bool indicating whether dynamic loss scaling is used.
                loss_scale_object = optimizer._loss_scale
                new_loss_scale = min(optimizer.loss_scale * (2 ** 4), MAX_LOSS_SCALE)
                logging.info(f'Forcefully increased {name} optimizer loss scale to {new_loss_scale}')
                # Taken from source code for LossScaleOptimizer
                loss_scale_object.counter.assign(0)
                loss_scale_object.current_loss_scale.assign(new_loss_scale)
    except:
        logging.error(f'Could not access required fields for loss scale optimizer in {tf.version.VERSION}. '
                      f'See source code of LossScaleOptimizer to fix the code accordingly.')


#----------------------------------------------------------------------------
# Images utils.

def generate_latents(batch_size: int, latents_size: int, dtype=tf.float32):
    return tf.random.normal(shape=[batch_size, latents_size], mean=0., stddev=1., dtype=dtype)


def restore_images(images):
    # Minimum OP doesn't support uint8
    images = tf.math.round((images + 1.0) * (255 / 2))
    images = tf.clip_by_value(tf.cast(images, dtype=tf.int32), 0, 255)
    images = tf.cast(images, dtype=tf.uint8)
    return images


def convert_outputs_to_images(net_outputs, target_single_image_size, hw_ratio=1, data_format=DEFAULT_DATA_FORMAT):
    # Note: should work for linear and tanh activation
    # 1. Adjust dynamic range of images
    x = restore_images(net_outputs)
    # 2. Optionally change data format
    validate_data_format(data_format)
    if data_format == NCHW_FORMAT:
        x = tf.transpose(x, toNHWC_AXIS)
    # 3. Optionally extract target images for wide dataset
    validate_hw_ratio(hw_ratio)
    x = extract_images(x, hw_ratio, NHWC_FORMAT)
    # 4. Resize images
    x = tf.image.resize(
        x,
        size=to_hw_size(target_single_image_size, hw_ratio),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return x


def extract_images(x, hw_ratio, data_format):
    if hw_ratio != 1:
        s = tf.shape(x)
        if data_format == NCHW_FORMAT:
            n, c, h, w = s[0], s[1], s[2], s[3]
            h = int(hw_ratio * float(h))  # h is tensor, so dtypes must match
            x = x[:, :, (w - h) // 2 : (w + h) // 2, :]
        else: # data_format == NHWC_FORMAT
            n, h, w, c = s[0], s[1], s[2], s[3]
            h = int(hw_ratio * float(h))  # h is tensor, so dtypes must match
            x = x[:, (w - h) // 2 : (w + h) // 2, :, :]
    return x


#----------------------------------------------------------------------------
# Logging utils.

def trace_vars(vars, title):
    # Ugly way to trace variables:
    # Python side-effects will only happen once, when func is traced
    log_debug_info = should_log_debug_info()
    if log_debug_info:
        os_name = platform.system()
        if OS_LINUX == os_name:
            logging.info('\n' + title)
            [logging.info(var.name) for var in vars]
        else:
            # Calling logging inside tf.function on Windows can cause error
            print('\n' + title)
            [print(var.name) for var in vars]
            print()


def trace_message(message):
    log_debug_info = should_log_debug_info()
    if log_debug_info:
        os_name = platform.system()
        if os_name == OS_LINUX:
            logging.info(message)
        else:
            # Calling logging inside tf.function on Windows can cause errors
            print(message)


def set_tf_logging(debug_mode=True):
    # https://github.com/tensorflow/tensorflow/issues/31870, see a comment by ziyigogogo on 28 Aug 2019
    # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information, answer by craymichael
    if debug_mode:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.get_logger().setLevel('INFO')
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)


def maybe_show_vars_stats(vars, message=None):
    if message is None:
        message = ''
    if should_log_debug_info():
        print('\n', message)
        for idx, var in enumerate(vars):
            mean = tf.math.reduce_mean(var).numpy()
            std = tf.math.reduce_std(var).numpy()
            print(f'{idx}) {var.name}: mean={mean:.3f}, std={std:.3f}')


def maybe_show_grads_stat(grads, vars, step, model_name):
    if should_log_debug_info():
        for grad, var in zip(grads, vars):
            nans = tf.math.count_nonzero(~tf.math.is_finite(grad))
            nums = tf.math.count_nonzero(tf.math.is_finite(grad))
            percent = tf.math.round(100 * nans / (nans + nums))
            tf.print(f'{model_name}, step = {step} {var.name}: n_nans is {nans} or {percent}%')


#----------------------------------------------------------------------------
# General utils.

def get_compute_dtype(use_mixed_precision):
    return mixed_precision.Policy('mixed_float16').compute_dtype if use_mixed_precision else 'float32'


def enable_mixed_precision_policy():
    mixed_precision.set_global_policy('mixed_float16')


def disable_mixed_precision_policy():
    mixed_precision.set_global_policy('float32')


def get_gpu_memory_usage():
    """
    Returns dict with info about memory usage (in Mb) by each GPU. {device_name: {dict}}
    """
    def format_device_name(n):
        # For TF 2.5 name is like '/device:GPU:0', for other versions this function might need changes
        return n.split('device')[1][1:]

    def to_mb_size(bytes_size):
        return int(bytes_size / (1024 * 1024))

    stats = {}
    for device in tf.config.experimental.list_logical_devices('GPU'):
        stats[format_device_name(device.name)] = {
            k: to_mb_size(v) for k, v in tf.config.experimental.get_memory_info(device.name).items()
        }
    return stats


# Linear interpolation
def lerp(a, b, t):
    """
    a multiplied by (1 - t)
    b multiplied by t
    """
    return a + (b - a) * t


def fp32(*values):
    if len(values) == 1:
        return tf.cast(values[0], tf.float32)
    return [tf.cast(v, tf.float32) for v in values]


def tf_bool(x):
    return tf.convert_to_tensor(x, dtype=tf.bool)


def run_model_on_batches(model, model_kwargs, inputs, batch_size):
    outputs = []
    s = tf.shape(inputs)
    n = s[0]
    for i in range(0, n, batch_size):
        start_idx = i
        end_idx = min(start_idx + batch_size, n)
        batch_inputs = inputs[start_idx : end_idx]
        outputs.append(model(batch_inputs, **model_kwargs))
    return tf.concat(outputs, axis=0)


def prepare_gpu(mode='auto', memory_limit=None):
    os_name = platform.system()
    os_message = f'\nScript is running on {os_name}, '

    # Note: change this number based on your GPU
    if memory_limit is None:
        memory_limit = 7600 # for real use. Larger values crash the app when starting (system after reboot, memory usage around 300 Mb)
        # memory_limit = 6000
    set_memory_growth = False
    set_memory_limit = False

    assert mode in ['auto', 'growth', 'limit']
    if mode == 'auto':
        if os_name == OS_LINUX:
            print(os_message + 'all memory is available for TensorFlow')
            return
            # print(os_message + 'memory growth option is used')
            # set_memory_growth = True
        elif os_name == OS_WIN:
            print(os_message + 'memory limit option is used')
            set_memory_limit = True
        else:
            print(
                os_message + f'GPU can only be configured for {OS_LINUX}|{OS_WIN}, '
                f'memory growth option is used'
            )
            set_memory_growth = True
    else:
        if mode == 'growth':
            set_memory_growth = True
        else:
            set_memory_limit = True

    physical_gpus = tf.config.experimental.list_physical_devices('GPU')

    if set_memory_limit:
        if len(physical_gpus) > 0:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ]
                    )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(
                    f'Physical GPUs: {len(physical_gpus)}, logical GPUs: {len(logical_gpus)}'
                )
                print(f'Set memory limit to {memory_limit} Mbs\n')
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('GPU is not available\n')

    if set_memory_growth:
        if len(physical_gpus) > 0:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Physical GPUs: {len(physical_gpus)} \nSet memory growth\n')
        else:
            print('GPU is not available\n')
