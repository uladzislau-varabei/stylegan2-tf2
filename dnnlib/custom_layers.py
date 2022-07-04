import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import mixed_precision

from config import Config as cfg
from dnnlib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d, DEFAULT_IMPL
from utils import validate_data_format
from tf_utils import FUNC_KEY, GAIN_KEY, ACTIVATION_FUNCS_DICT, FP32_ACTIVATIONS,\
    DEFAULT_DATA_FORMAT, NCHW_FORMAT, NHWC_FORMAT, RANDOMIZE_NOISE_VAR_NAME, RANDOM_NOISE_WEIGHT, HE_GAIN,\
    PER_LAYER_COMPILATION, clip_by_value_preserve_gradient, lerp


LRMUL = 1.

WEIGHTS_NAME = 'weights'
BIASES_NAME = 'biases'

DEFAULT_USE_WSCALE       = cfg.DEFAULT_USE_WSCALE
DEFAULT_TRUNCATE_WEIGHTS = cfg.DEFAULT_TRUNCATE_WEIGHTS
DEFAULT_DTYPE            = cfg.DEFAULT_DTYPE
DEFAULT_USE_XLA          = cfg.DEFAULT_USE_XLA


#----------------------------------------------------------------------------
# Utils.

global_seed = 0
biases_init = tf.zeros_initializer()


def weights_init(std):
    return tf.random_normal_initializer(stddev=std, seed=global_seed)


def truncated_weights_init(std):
    return tf.initializers.TruncatedNormal(stddev=std, seed=global_seed)


def select_initializer(truncate_weights, std):
    if truncate_weights:
        return truncated_weights_init(std)
    else:
        return weights_init(std)


def weights_std(gain, fan_in):
    # He/LeCun init
    return gain / np.sqrt(fan_in)


# Equalized learning rate and custom learning rate multiplier
def weights_coeffs(use_wscale, std, lrmul):
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = std * lrmul
    else:
        init_std = std / lrmul
        runtime_coef = lrmul
    return init_std, runtime_coef


# Remove layer name from name scope
# To be used as a second level of scoping in build with
# ... tf.name_scope(clean_name_scope(scope)) as final_scope: ...
def clean_name_scope(name_scope):
    return name_scope.split('/', 1)[1] if '/' in name_scope else name_scope


# Make layer name to structure graph in tensorboard
def make_layer_name(input_name, input_scope, class_scope):
    return input_name if input_name is not None else input_scope + class_scope


#----------------------------------------------------------------------------
# 2D convolution op with optional upsampling, downsampling and padding.

def validate_conv_kernel(w):
    kernel = w.shape[0]
    assert w.shape[1] == kernel
    assert kernel >= 1 and kernel % 2 == 1


def conv2d_op(x, w, up=False, down=False, resample_kernel=None, padding=0, strides=None, impl=DEFAULT_IMPL, data_format=DEFAULT_DATA_FORMAT):
    assert not (up and down)
    validate_data_format(data_format)
    validate_conv_kernel(w)
    if up:
        x = upsample_conv_2d(x, w, k=resample_kernel, padding=padding, data_format=data_format, impl=impl)
    elif down:
        x = conv_downsample_2d(x, w, k=resample_kernel, padding=padding, data_format=data_format, impl=impl)
    else:
        if strides is None:
            strides = [1, 1, 1, 1]
        padding_mode = {0: 'SAME', -(w.shape[0] // 2): 'VALID'}[padding]
        x = tf.nn.conv2d(x, w, data_format=data_format, strides=strides, padding=padding_mode)
    return x


#----------------------------------------------------------------------------
# Layers.

class AdvConv2d(Layer):

    def __init__(self, fmaps, kernel_size, stride=1, up=False, down=False, resample_kernel=None, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA, impl=DEFAULT_IMPL,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'AdvConv2d')
        super(AdvConv2d, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.stride = stride
        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.impl = impl
        self.scope = scope
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, self.stride, self.stride]
        else: # self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, self.stride, self.stride, 1]

        self.wshape = [self.kernel_size, self.kernel_size, self.channels_in, self.fmaps]
        self.fan_in = np.prod(self.wshape[:-1])
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=self.wshape,
                initializer=initializer,
                trainable=True
            )

    def call(self, x, *args, **kwargs):
        return conv2d_op(
            x, self.runtime_coef * self.w, up=self.up, down=self.down, resample_kernel=self.resample_kernel,
            strides=self.strides, impl=self.impl, data_format=self.data_format
        )


class ModulatedConv2d(Layer):
    def __init__(self, fmaps, kernel_size, stride=1,
                 up=False, down=False, demodulate=True, resample_kernel=None, fused_modconv=False, use_bias=True, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA, impl=DEFAULT_IMPL,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'ModulatedConv2d')
        super(ModulatedConv2d, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        assert kernel_size >= 1 and kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.stride = stride
        assert not (up and down)
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.resample_kernel = resample_kernel
        self.fused_modconv = fused_modconv
        self.use_bias = use_bias
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.impl = impl
        self.use_fp16 = (self._dtype_policy.compute_dtype != 'float32')
        self.use_fp16_normalization = (self.use_fp16) and (not self.fused_modconv) and (self.demodulate)
        self.epsilon = 1e-4 if self.use_fp16 else 1e-8
        self.scope = scope + 'ModConv2d/'
        self.style_scope = scope + 'StyleModFC/'
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        x_shape, style_shape = input_shape

        if self.data_format == NCHW_FORMAT:
            self.channels_in = x_shape[1]
            self.strides = [1, 1, self.stride, self.stride]
            self.h_axis = 2
            self.w_axis = 3
        else:  # self.data_format == NHWC_FORMAT:
            self.channels_in = x_shape[-1]
            self.strides = [1, self.stride, self.stride, 1]
            self.h_axis = 1
            self.w_axis = 2

        self.wshape = [self.kernel_size, self.kernel_size, self.channels_in, self.fmaps]
        self.fan_in = np.prod(self.wshape[:-1])
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=self.wshape,
                initializer=initializer,
                trainable=True
            )

        self.fc = AdvFullyConnected(
            units=self.channels_in, gain=1,
            use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
            dtype=self._dtype_policy, use_xla=self.use_xla,
            data_format=self.data_format, scope=self.style_scope
        )
        if self.use_bias:
            self.bias = Bias(
                dtype=self._dtype_policy, use_xla=self.use_xla, scope=self.style_scope, data_format=self.data_format
            )

        self.fp16_normalization_shape_scale = tf.cast(tf.sqrt(1 / tf.reduce_prod(self.wshape[:-1])), self._dtype_policy.compute_dtype)

    def maybe_apply_bias(self, x):
        return self.bias(x) + 1 if self.use_bias else x # [BI] Add bias (initially 1).

    def call(self, inputs, *args, **kwargs):
        # inputs: x - main tensor, y - dlatents
        x, y = inputs
        w = self.w * self.runtime_coef
        if self.use_fp16_normalization:
            w *= self.fp16_normalization_shape_scale / tf.reduce_max(tf.abs(w), axis=[0, 1, 2]) # Pre-normalize to avoid float16 overflow.
        ww = w[tf.newaxis] # [BkkIO] Introduce minibatch dimension.

        # Modulate.
        s = self.fc(y)
        s = self.maybe_apply_bias(s) # [BI] Add bias (initially 1).
        if self.use_fp16_normalization:
            s *= 1 / tf.reduce_max(tf.abs(s)) # Pre-normalize to avoid fp16 overflow.
        ww *= s[:, tf.newaxis, tf.newaxis, :, tf.newaxis] # [BkkIO] Scale input feature maps.

        # Demodulate.
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1, 2, 3]) + self.epsilon) # [BO] Scaling factor.
            ww *= d[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        # Reshape/scale input.
        if self.fused_modconv:
            # TODO: check how it works with NHWC data format
            x = tf.reshape(x, [1, -1, x.shape[self.h_axis], x.shape[self.w_axis]]) # Fused => reshape minibatch to convolution groups.
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
        else:
            x *= s[:, :, tf.newaxis, tf.newaxis] # [BIhw] Not fused => scale input activations.

        # 2D convolution.
        x = conv2d_op(
            x, w, up=self.up, down=self.down, resample_kernel=self.resample_kernel, strides=self.strides, impl=self.impl, data_format=self.data_format
        )

        # Reshape/scale output.
        if self.fused_modconv:
            if self.data_format == NCHW_FORMAT:
                output_shape = [-1, self.fmaps, x.shape[self.h_axis], x.shape[self.w_axis]]
            else: # if self.data_format == NHWC_FORMAT:
                output_shape = [-1, x.shape[self.h_axis], x.shape[self.w_axis], self.fmaps]
            x = tf.reshape(x, output_shape) # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x *= d[:, :, tf.newaxis, tf.newaxis] # [BOhw] Not fused => scale output activations.
        return x


class AdvFullyConnected(Layer):

    def __init__(self, units, gain=HE_GAIN,
                 use_wscale=DEFAULT_USE_WSCALE, lrmul=LRMUL, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'AdvFC')
        super(AdvFullyConnected, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.units = units
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        self.scope = scope
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        self.fan_in = np.prod(input_shape[1:])
        self.std = weights_std(self.gain, self.fan_in)
        self.init_std, self.runtime_coef = weights_coeffs(self.use_wscale, self.std, self.lrmul)

        initializer = select_initializer(self.truncate_weights, self.init_std)

        with tf.name_scope(self.scope):
            self.w = self.add_weight(
                name=WEIGHTS_NAME,
                shape=[self.fan_in, self.units],
                initializer=initializer,
                trainable=True
            )

    def call(self, x, *args, **kwargs):
        # Use inplace ops
        return tf.linalg.matmul(
            tf.reshape(x, [-1, self.fan_in]) if len(x.shape) > 2 else x,
            self.runtime_coef * self.w
        )


class Bias(Layer):

    def __init__(self, lrmul=LRMUL, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Bias')
        super(Bias, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.lrmul = lrmul
        self.use_xla = use_xla
        self.scope = scope
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        self.is_linear_bias = len(input_shape) == 2

        if self.is_linear_bias:
            self.units = input_shape[1]
        else:
            if self.data_format == NCHW_FORMAT:
                self.bias_target_shape = [1, -1, 1, 1]
                self.units = input_shape[1]
            else: # self.data_format == NHWC_FORMAT:
                self.bias_target_shape = [1, 1, 1, -1]
                self.units = input_shape[-1]

        with tf.name_scope(self.scope):
            self.b = self.add_weight(
                name=BIASES_NAME,
                shape=[self.units],
                initializer=biases_init,
                trainable=True
            )

    def call(self, x, *args, **kwargs):
        # Note: keep reshaping to allow easy weights decay between cpu and gpu models
        return x + self.lrmul * (self.b if self.is_linear_bias else tf.reshape(self.b, self.bias_target_shape))


class Upscale2d(Layer):

    def __init__(self, factor, resample_kernel=None, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Upscale2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.factor = factor
        self.resample_kernel = resample_kernel
        self.use_xla = use_xla
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def call(self, x, *args, **kwargs):
        return upsample_2d(x, k=self.resample_kernel, factor=self.factor, data_format=self.data_format)


class Downscale2d(Layer):

    def __init__(self, factor, resample_kernel=None, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Downscale2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.factor = factor
        self.resample_kernel = resample_kernel
        self.use_xla = use_xla
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def call(self, x, *args, **kwargs):
        return downsample_2d(x, k=self.resample_kernel, factor=self.factor, data_format=self.data_format)


class PixelNorm(Layer):

    def __init__(self, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'PixelNorm')
        super(PixelNorm, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.epsilon = 1e-8 if self._dtype_policy.compute_dtype == 'float32' else 1e-4
        self.use_xla = use_xla
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        # Layer might also be used to normalize latents, which hase shape [batch, channels]
        if len(input_shape) == 2:
            self.channel_axis = 1
        else:
            if self.data_format == NCHW_FORMAT:
                self.channel_axis = 1
            else:  # self.data_format == NHWC_FORMAT:
                self.channel_axis = 3

    def call(self, x, *args, **kwargs):
        x = tf.cast(x, tf.float32)
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=self.channel_axis, keepdims=True) + self.epsilon
        )


class Noise(Layer):

    def __init__(self, randomize_noise, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = name if name is not None else scope + 'Noise'
        super(Noise, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.randomize_noise = randomize_noise
        self.use_xla = use_xla
        self.scope = scope #+ 'Noise'
        self.tf_zero = tf.constant(0.0, dtype=self._dtype_policy.compute_dtype, name='zero')
        if PER_LAYER_COMPILATION:
            # XLA doesn't seem to work with tf.cond, so compile this layer explicitly with tf.function
            # (doesn't work when model call is wrapped with jit_compile) or use lerp
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.noise_tail_shape = [1, input_shape[2], input_shape[3]]
        else: # self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.noise_tail_shape = [input_shape[1], input_shape[2], 1]

        with tf.name_scope(self.scope) as scope:
            #with tf.name_scope(clean_name_scope(scope)) as final_scope:
            self.w = self.add_weight(
                name='noise_strength',
                shape=[],
                initializer=tf.zeros_initializer(),
                trainable=True
            )
            # Always create non-random noise to allow easy testing
            # TODO: think how to handle batch dim (when building layer input_shape[0] is None)
            self.const_noise = self.add_weight(
                name='const_noise',
                shape=[1] + self.noise_tail_shape,
                initializer=tf.random_normal_initializer(),
                trainable=False
            )
            # Add weight to control weight for random noise
            self.random_noise_weight = self.add_weight(
                name=RANDOM_NOISE_WEIGHT,
                initializer=tf.constant_initializer(1. if self.randomize_noise else 0.),
                trainable=False,
                dtype=self._dtype_policy.compute_dtype
            )

    def call(self, x, *args, **kwargs):
        # One can change layer weights (see random_noise_weight) to switch between random and non random noise
        rand_noise = tf.random.normal([tf.shape(x)[0]] + self.noise_tail_shape, dtype=self._dtype_policy.compute_dtype)
        const_noise = tf.tile(self.const_noise, [tf.shape(x)[0], 1, 1, 1])
        noise = lerp(const_noise, rand_noise, self.random_noise_weight)
        return x + noise * self.w


class Const(Layer):

    def __init__(self, channel_size, dtype=DEFAULT_DTYPE,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Const')
        super(Const, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.channel_size = channel_size
        # Taken from the original implementation
        self.hw_size = 4
        self.scope = scope

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.shape = [1, self.channel_size, self.hw_size, self.hw_size]
        else: # self.data_format == NHWC_FORMAT:
            self.shape = [1, self.hw_size, self.hw_size, self.channel_size]

        with tf.name_scope(self.scope):
            self.const_input = self.add_weight(
                name='const',
                shape=self.shape,
                initializer=tf.random_normal_initializer(stddev=1., seed=global_seed),
                trainable=True
            )

    def call(self, x, *args, **kwargs):
        # x - latents
        return tf.tile(self.const_input, [tf.shape(x)[0], 1, 1, 1])


class MinibatchStdDev(Layer):

    def __init__(self, group_size=4, num_new_features=1, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'MinibatchStddev')
        super(MinibatchStdDev, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.group_size = group_size
        self.num_new_features = num_new_features
        self.use_xla = use_xla
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def call(self, x, *args, **kwargs):
        if self.data_format == NCHW_FORMAT:
            _, c, h, w = x.shape
            n = tf.shape(x)[0]
            group_size = tf.math.minimum(self.group_size, n)     # Minibatch must be divisible or smaller than batch size
            # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c
            y = tf.reshape(x, [group_size, -1, self.num_new_features, c // self.num_new_features, h, w])
            y = tf.cast(y, tf.float32)                           # [GMncHW] Cast to fp32
            y -= tf.reduce_mean(y, axis=0, keepdims=True)        # [GMncHW] Subtract mean over group
            y = tf.reduce_mean(tf.square(y), axis=0)             # [MncHW] Variance over group
            y = tf.sqrt(y + 1e-8)                                # [MncHW] Stddev over group
            y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True) # [Mn111] Average over fmaps and pixels
            y = tf.reduce_mean(y, axis=[2])                      # [Mn11]
            y = tf.cast(y, x.dtype)                              # [Mn11] Cast back to original dtype
            y = tf.tile(y, [group_size, 1, h, w])                # [NnHW] Replicate over group and pixels
            return tf.concat([x, y], axis=1)
        else: # self.data_format == NHWC_FORMAT:
            _, h, w, c = x.shape
            n = tf.shape(x)[0]
            group_size = tf.math.minimum(self.group_size, n)     # Minibatch must be divisible or smaller than batch size
            # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c
            y = tf.reshape(x, [group_size, -1, h, w, self.num_new_features, c // self.num_new_features])
            y = tf.cast(y, tf.float32)                           # [GMHWnc] Cast to fp32
            y -= tf.reduce_mean(y, axis=0, keepdims=True)        # [GMHWnc] Subtract mean over group
            y = tf.reduce_mean(tf.square(y), axis=0)             # [MHWnc] Variance over group
            y = tf.sqrt(y + 1e-8)                                # [MHWnc] Stddev over group
            y = tf.reduce_mean(y, axis=[1, 2, 4], keepdims=True) # [M11n1] Average over fmaps and pixels
            y = tf.reduce_mean(y, axis=[4])                      # [M11n]
            y = tf.cast(y, x.dtype)                              # [M11n] Cast back to original dtype
            y = tf.tile(y, [group_size, h, w, 1])                # [NHWn] Replicate over group and pixels
            return tf.concat([x, y], axis=3)


class Activation(Layer):

    def __init__(self, act_name=None, clamp=None, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Activation')
        super(Activation, self).__init__(dtype=dtype, name=layer_name)
        self.act_name = act_name
        self.fp32_act = (act_name in FP32_ACTIVATIONS) and (self._dtype_policy.compute_dtype != 'float32')
        self.fp32_act_gain = self.fp32_act or (self._dtype_policy.compute_dtype == 'float32')
        if act_name.lower() in ACTIVATION_FUNCS_DICT.keys():
            act = ACTIVATION_FUNCS_DICT[act_name.lower()]
            self.act_func = act[FUNC_KEY]
            self.act_gain = tf.constant(act[GAIN_KEY], dtype=('float32' if self.fp32_act_gain else 'float16'))
        else:
            assert False, f"Activation '{act_name}' is not supported. See ACTIVATION_FUNS_DICT"
        if clamp is not None:
            assert clamp > 0, 'Clamp should be greater than 0'
        self.clamp = clamp
        self.use_xla = use_xla
        self.scope = scope
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def call(self, x, *args, **kwargs):
        x = self.act_gain * self.act_func(tf.cast(x, tf.float32) if self.fp32_act else x)

        if self.clamp is not None:
            # Note: for some reasons when training with mixed precision (all fine for fp32)
            # Grappler optimizer raises an error for D (but not G) network (layout failed) if values all clipped this way:
            #   "x = tf.clip_by_value(x, -self.clamp, self.clamp)" -- doesn't work correctly
            # The solution is to transpose inputs to NHWC format, clip them and transpose back to NCHW.
            # Problem exists at least for NCHW format, which is the one used by GPU.

            # To disable layout optimization see: https://github.com/tensorflow/tensorflow/issues/36901 (answer by ninnghazad).
            # More about graph optimization: www.tensorflow.org/guide/graph_optimization (see layout optimizer).
            # It should be possible to disable optimization only in current place using context manager (see tf link above).

            # This approach to clipping solves all issues, however, is it correct to preserve gradient for clipped values?
            # The official implementation uses tf.clip_by_value.
            x = clip_by_value_preserve_gradient(x, -self.clamp, self.clamp)

        return x


# Fused bias + activation.
# Custom cuda implementation is faster and uses less memory than performing the operations separately.
# Maybe XLA helps at least a bit to achieve a similar effect.
class FusedBiasAct(Layer):

    def __init__(self, use_bias=True, act_name=None, lrmul=LRMUL, clamp=None,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, scope='', name=None):
        layer_name = make_layer_name(name, scope, 'Fused_Bias_Act')
        super(FusedBiasAct, self).__init__(dtype=dtype, name=layer_name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.use_bias = use_bias
        self.act_name = act_name
        self.fp32_act = (act_name in FP32_ACTIVATIONS) and (self._dtype_policy.compute_dtype != 'float32')
        self.fp32_act_gain = self.fp32_act or (self._dtype_policy.compute_dtype == 'float32')
        if act_name.lower() in ACTIVATION_FUNCS_DICT.keys():
            act = ACTIVATION_FUNCS_DICT[act_name.lower()]
            self.act_func = act[FUNC_KEY]
            self.act_gain = tf.constant(act[GAIN_KEY], dtype=('float32' if self.fp32_act_gain else 'float16'))
        else:
            assert False, f"Activation '{act_name}' is not supported. See ACTIVATION_FUNS_DICT"
        self.lrmul = lrmul
        if clamp is not None:
            assert clamp > 0, 'Clamp should be greater than 0'
        self.clamp = clamp
        self.use_xla = use_xla
        self.scope = scope
        if PER_LAYER_COMPILATION:
            self.call = tf.function(self.call, jit_compile=self.use_xla)

    def build(self, input_shape):
        self.is_linear_bias = len(input_shape) == 2

        if self.is_linear_bias:
            self.units = input_shape[1]
        else:
            if self.data_format == NCHW_FORMAT:
                self.bias_target_shape = [1, -1, 1, 1]
                self.units = input_shape[1]
            else: # self.data_format == NHWC_FORMAT:
                self.bias_target_shape = [1, 1, 1, -1]
                self.units = input_shape[-1]

        if self.use_bias:
            with tf.name_scope(self.scope):
                self.b = self.add_weight(
                    name=BIASES_NAME,
                    shape=[self.units],
                    initializer=biases_init,
                    trainable=True
                )

    def call(self, x, *args, **kwargs):
        if self.use_bias:
            # Note: keep reshaping to allow easy weights decay between cpu and gpu models
            x += self.lrmul * (self.b if self.is_linear_bias else tf.reshape(self.b, self.bias_target_shape))

        x = self.act_gain * self.act_func(tf.cast(x, tf.float32) if self.fp32_act else x)

        if self.clamp is not None:
            # Note: for some reasons when training with mixed precision (all fine for fp32)
            # Grappler optimizer raises an error for D (but not G) network (layout failed) if values all clipped this way:
            #   "x = tf.clip_by_value(x, -self.clamp, self.clamp)" -- doesn't work correctly
            # The solution is to transpose inputs to NHWC format, clip them and transpose back to NCHW.
            # Problem exists at least for NCHW format, which is the one used by GPU.

            # To disable layout optimization see: https://github.com/tensorflow/tensorflow/issues/36901 (answer by ninnghazad).
            # More about graph optimization: www.tensorflow.org/guide/graph_optimization (see layout optimizer).
            # It should be possible to disable optimization only in current place using context manager (see tf link above).

            # This approach to clipping solves all issues, however, is it correct to preserve gradient for clipped values?
            # The official implementation uses tf.clip_by_value.
            x = clip_by_value_preserve_gradient(x, -self.clamp, self.clamp)

        return x


#----------------------------------------------------------------------------
# Layers as functions.

class LAYERS:
    conv2d           = 'conv2d'
    modulated_conv2d = 'modulated_conv2d'
    fully_connected  = 'fully_connected'
    bias             = 'bias'
    act              = 'act'
    const            = 'const'
    noise            = 'noise'
    pixel_norm       = 'pixel_norm'
    downscale2d      = 'downscale2d'
    upscale2d        = 'upscale2d'
    minibatch_stddev = 'minibatch_stddev'
    resnet_merge     = 'resnet_merge'
    skip_merge       = 'skip_merge'


def layer_dtype(layer_type, use_fp16=None, act_name=None, config=None):
    if use_fp16 is not None:
        use_mixed_precision = use_fp16
    else:
        use_mixed_precision = config.get(cfg.USE_MIXED_PRECISION, cfg.DEFAULT_USE_MIXED_PRECISION)

    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        act_dtype = 'float32' if act_name in FP32_ACTIVATIONS else policy
        compute_dtype = policy.compute_dtype
    else:
        policy = 'float32'
        act_dtype = 'float32'
        compute_dtype = 'float32'

    if layer_type in [
        LAYERS.conv2d, LAYERS.modulated_conv2d, LAYERS.fully_connected, LAYERS.bias, LAYERS.noise, LAYERS.const
    ]:
        return policy
    elif layer_type == LAYERS.act:
        return act_dtype
    elif layer_type in [
        LAYERS.pixel_norm, LAYERS.upscale2d, LAYERS.downscale2d,
        LAYERS.minibatch_stddev, LAYERS.resnet_merge, LAYERS.skip_merge
    ]:
        return compute_dtype
    else:
        assert False, 'Unknown layer type'


def conv2d_layer(x, fmaps, kernel_size, gain, lrmul=LRMUL,
                 up=False, down=False, use_fp16=None, scope='', config=None):
    assert not (up and down)

    resample_kernel  = config.get(cfg.RESAMPLE_KERNEL, cfg.DEFAULT_RESAMPLE_KERNEL)
    use_wscale       = config.get(cfg.USE_WSCALE, cfg.DEFAULT_USE_WSCALE)
    truncate_weights = config.get(cfg.TRUNCATE_WEIGHTS, cfg.DEFAULT_TRUNCATE_WEIGHTS)
    use_xla          = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format      = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy           = layer_dtype(LAYERS.conv2d, use_fp16=use_fp16)

    layer_kwargs = {
        'fmaps'           : fmaps,
        'kernel_size'     : kernel_size,
        'up'              : up,
        'down'            : down,
        'resample_kernel' : resample_kernel,
        'gain'            : gain,
        'use_wscale'      : use_wscale,
        'lrmul'           : lrmul,
        'truncate_weights': truncate_weights,
        'dtype'           : policy,
        'use_xla'         : use_xla,
        'data_format'     : data_format,
        'scope'           : scope
    }
    return AdvConv2d(**layer_kwargs)(x)


def modulated_conv2d_layer(x, dlatents, fmaps, kernel_size, up=False, down=False, demodulate=True,
                           gain=HE_GAIN, lrmul=LRMUL, use_fp16=None, scope='', config=None):
    assert not (up and down)

    resample_kernel  = config.get(cfg.RESAMPLE_KERNEL, cfg.DEFAULT_RESAMPLE_KERNEL)
    fused_modconv    = config.get(cfg.FUSED_MODCONV, cfg.DEFAULT_FUSED_MODCONV)
    use_bias         = config.get(cfg.USE_BIAS, cfg.DEFAULT_USE_BIAS)
    use_wscale       = config.get(cfg.USE_WSCALE, cfg.DEFAULT_USE_WSCALE)
    truncate_weights = config.get(cfg.TRUNCATE_WEIGHTS, cfg.DEFAULT_TRUNCATE_WEIGHTS)
    use_xla          = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format      = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy           = layer_dtype(LAYERS.modulated_conv2d, use_fp16=use_fp16)

    layer_kwargs = {
        'fmaps'           : fmaps,
        'kernel_size'     : kernel_size,
        'up'              : up,
        'down'            : down,
        'demodulate'      : demodulate,
        'resample_kernel' : resample_kernel,
        'fused_modconv'   : fused_modconv,
        'use_bias'        : use_bias,
        'gain'            : gain,
        'use_wscale'      : use_wscale,
        'lrmul'           : lrmul,
        'truncate_weights': truncate_weights,
        'dtype'           : policy,
        'use_xla'         : use_xla,
        'data_format'     : data_format,
        'scope'           : scope
    }
    return ModulatedConv2d(**layer_kwargs)([x, dlatents])


def fully_connected_layer(x, units, gain, lrmul=LRMUL, use_fp16=None, scope='', config=None):
    use_wscale       = config.get(cfg.USE_WSCALE, cfg.DEFAULT_USE_WSCALE)
    truncate_weights = config.get(cfg.TRUNCATE_WEIGHTS, cfg.DEFAULT_TRUNCATE_WEIGHTS)
    use_xla          = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format      = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy           = layer_dtype(LAYERS.fully_connected, use_fp16=use_fp16)
    return AdvFullyConnected(
        units=units, gain=gain,
        use_wscale=use_wscale, lrmul=lrmul, truncate_weights=truncate_weights,
        dtype=policy, use_xla=use_xla, data_format=data_format, scope=scope
    )(x)


def bias_layer(x, lrmul=LRMUL, use_fp16=None, scope='', config=None):
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(LAYERS.bias, use_fp16=use_fp16)
    return Bias(lrmul=lrmul, dtype=policy, use_xla=use_xla, data_format=data_format, scope=scope)(x)


def act_layer(x, act_name, clamp=None, use_fp16=None, scope='', config=None):
    # No activation => early exit
    if act_name.lower() == 'linear':
        return x
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    dtype = layer_dtype(LAYERS.act, use_fp16=use_fp16, act_name=act_name)
    return Activation(act_name=act_name, clamp=clamp, dtype=dtype, use_xla=use_xla, scope=scope)(x)


def fused_bias_act_layer(x, act_name, use_bias, lrmul=LRMUL, clamp=None, use_fp16=None, scope='', config=None):
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(LAYERS.bias, use_fp16=use_fp16)
    return FusedBiasAct(
        use_bias=use_bias, act_name=act_name, lrmul=lrmul, clamp=clamp,
        dtype=policy, use_xla=use_xla, data_format=data_format, scope=scope
    )(x)


def bias_act_layer(x, act_name, use_bias, lrmul=LRMUL, clamp=None, use_fp16=None, scope='', config=None):
    # 1. Apply bias
    if use_bias:
        x = bias_layer(x, lrmul, use_fp16=use_fp16, scope=scope, config=config)
    # 2. Apply activation (act + gain + clamp)
    x = act_layer(x, act_name, clamp=clamp, use_fp16=use_fp16, scope=scope, config=config)
    return x


def const_layer(x, channel_size, use_fp16=None, scope='', config=None):
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(LAYERS.const, use_fp16=use_fp16)
    return Const(channel_size=channel_size, dtype=policy, data_format=data_format, scope=scope)(x)


def noise_layer(x, use_fp16=None, scope='', config=None):
    randomize_noise = config.get(cfg.RANDOMIZE_NOISE, cfg.DEFAULT_RANDOMIZE_NOISE)
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    policy = layer_dtype(LAYERS.noise, use_fp16=use_fp16)
    return Noise(randomize_noise=randomize_noise, dtype=policy, use_xla=use_xla, data_format=data_format, scope=scope)(x)


def pixel_norm_layer(x, use_fp16=None, scope='', config=None):
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(LAYERS.pixel_norm, use_fp16=use_fp16)
    return PixelNorm(dtype=dtype, use_xla=use_xla, data_format=data_format, scope=scope)(x)


def downscale2d_layer(x, factor, use_fp16=None, config=None, name=None):
    resample_kernel = config.get(cfg.RESAMPLE_KERNEL, cfg.DEFAULT_RESAMPLE_KERNEL)
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(LAYERS.downscale2d, use_fp16=use_fp16)
    return Downscale2d(
        factor=factor, resample_kernel=resample_kernel, dtype=dtype, use_xla=use_xla, data_format=data_format, name=name
    )(x)


def upscale2d_layer(x, factor, use_fp16=None, config=None, name=None):
    resample_kernel = config.get(cfg.RESAMPLE_KERNEL, cfg.DEFAULT_RESAMPLE_KERNEL)
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(LAYERS.upscale2d, use_fp16=use_fp16)
    return Upscale2d(
        factor=factor, resample_kernel=resample_kernel, dtype=dtype, use_xla=use_xla, data_format=data_format, name=name
    )(x)


def minibatch_stddev_layer(x, use_fp16=None, scope='', config=None):
    group_size = config.get(cfg.MBSTD_GROUP_SIZE, 4)
    num_new_features = config.get(cfg.MBSTD_NUM_FEATURES, cfg.DEFAULT_MBSTD_NUM_FEATURES)
    use_xla = config.get(cfg.USE_XLA, cfg.DEFAULT_USE_XLA)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    dtype = layer_dtype(LAYERS.minibatch_stddev, use_fp16=use_fp16)
    return MinibatchStdDev(
        group_size=group_size, num_new_features=num_new_features,
        dtype=dtype, use_xla=use_xla, data_format=data_format, scope=scope
    )(x)


def resnet_merge_layer(x, y, use_fp16=None, name=None):
    def merge_func(a, dtype):
        c = tf.cast(1 / tf.sqrt(2.), dtype)
        return (a[0] + a[1]) * c
    dtype = layer_dtype(LAYERS.resnet_merge, use_fp16=use_fp16)
    layer = tf.keras.layers.Lambda(merge_func, arguments={'dtype': dtype}, dtype=dtype, name=name)
    return layer([x, y])


def skip_merge_layer(x, y, use_fp16=None, name=None):
    def merge_func(a):
        return a[0] + a[1]
    dtype = layer_dtype(LAYERS.skip_merge, use_fp16=use_fp16)
    layer = tf.keras.layers.Lambda(merge_func, dtype=dtype, name=name)
    return layer([x, y])
