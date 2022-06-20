import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dropout, Conv2D, Permute
from tensorflow.keras.applications.vgg16 import VGG16

from tf_utils import disable_mixed_precision_policy


def preprocess_image(image):
    orig_dtype = image.dtype
    image = tf.cast(image, tf.float32)

    factor = 255.0 / 2.0
    center = 1.0
    # Input image format is NHWC, so these values are broadcasted
    scale = tf.constant([0.458, 0.448, 0.450])[tf.newaxis, tf.newaxis, tf.newaxis, :]
    shift = tf.constant([-0.030, -0.088, -0.188])[tf.newaxis, tf.newaxis, tf.newaxis, :]

    image = image / factor - center  # [0.0 ~ 255.0] -> [-1.0 ~ 1.0]
    image = (image - shift) / scale

    image = tf.cast(image, orig_dtype)
    return image


def vgg_perceptual_metric_model(image_size, model_path):
    # initialize all models
    net = vgg_model(image_size)
    # Linear model uses lots of operations which should be executed in fp32, so disable mixed precision evaluation here
    disable_mixed_precision_policy()
    lin = linear_model(image_size)

    # Note: originally these 2 layers were built with dtype fp32
    # merge two model
    input1 = Input(shape=(image_size[0], image_size[1], 3), name='input1')
    input2 = Input(shape=(image_size[0], image_size[1], 3), name='input2')

    # preprocess input images
    net_out1 = Lambda(lambda x: preprocess_image(x))(input1)
    net_out2 = Lambda(lambda x: preprocess_image(x))(input2)

    # run vgg model first
    net_out1 = net(net_out1)
    net_out2 = net(net_out2)

    # nhwc -> nchw (after these layers outputs have fp32 dtype, as mixed precision is disabled after vgg model)
    net_out1 = [Permute(dims=(3, 1, 2))(t) for t in net_out1]
    net_out2 = [Permute(dims=(3, 1, 2))(t) for t in net_out2]

    # normalize
    normalize_tensor = lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))
    net_out1 = [Lambda(lambda x: normalize_tensor(x))(t) for t in net_out1]
    net_out2 = [Lambda(lambda x: normalize_tensor(x))(t) for t in net_out2]

    # subtract
    diffs = [Lambda(lambda x: tf.square(x[0] - x[1]))([t1, t2]) for t1, t2 in zip(net_out1, net_out2)]

    # run on learned linear model
    lin_out = lin(diffs)

    # take spatial average: list([N, 1], [N, 1], [N, 1], [N, 1], [N, 1])
    lin_out = [Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3], keepdims=False))(t) for t in lin_out]

    # take sum of all layers: [N, 1]
    lin_out = Lambda(lambda x: tf.add_n(x))(lin_out)

    # squeeze: [N, ]
    lin_out = Lambda(lambda x: tf.squeeze(x, axis=-1))(lin_out)

    final_model = Model(inputs=[input1, input2], outputs=lin_out)
    final_model.load_weights(model_path)
    return final_model


def vgg_model(image_size):
    # (None, 64, 64, 64)
    # (None, 32, 32, 128)
    # (None, 16, 16, 256)
    # (None, 8, 8, 512)
    # (None, 4, 4, 512)
    layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    vgg16 = VGG16(include_top=False, weights=None, input_shape=(image_size[0], image_size[1], 3))

    vgg16_output_layers = [l.output for l in vgg16.layers if l.name in layers]
    model = Model(vgg16.input, vgg16_output_layers, name='perceptual_model')
    return model


def linear_model(input_image_size):
    vgg_channels = [64, 128, 256, 512, 512]
    inputs, outputs = [], []
    for ii, channel in enumerate(vgg_channels):
        h_size = input_image_size[0] // (2 ** ii)
        w_size = input_image_size[1] // (2 ** ii)

        # Note: keep fp32 dtype to avoid numerical issues later with upcoming operations
        model_input = Input(shape=(channel, h_size, w_size), dtype=tf.float32)
        model_output = Dropout(rate=0.5, dtype=tf.float32)(model_input)
        model_output = Conv2D(filters=1, kernel_size=1, strides=1, use_bias=False,
                              data_format='channels_first', dtype=tf.float32, name=f'lin_{ii}')(model_output)
        inputs.append(model_input)
        outputs.append(model_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear_model')
    return model
