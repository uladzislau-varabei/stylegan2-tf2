import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

def my_func():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

def my_func2():
    from dnnlib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d

    inC = 128
    dtype = 'float16'
    x = tf.random.normal(shape=[8, inC, 256, 256], dtype=dtype)
    k = [1, 3, 3, 1]
    w = tf.random.normal(shape=[3, 3, inC, 128], dtype=dtype)

    # impl = 'ref'
    impl = 'custom_grad'

    with tf.GradientTape() as tape:
        tape.watch(w)
        up_res = upsample_conv_2d(x, w, k, data_format='NCHW', impl=impl)
    up_grads = tape.gradient(up_res, w)
    # print(f'up_grads: {up_grads}')

    print(f'Finished upsample with impl={impl} and dtype={dtype}')

    with tf.GradientTape() as tape:
        tape.watch(w)
        down_res = conv_downsample_2d(x, w, k, data_format='NCHW', impl=impl)
    down_grads = tape.gradient(down_res, w)
    # print(f'down_grads: {down_grads}')

    print(f'Finished downsample with impl={impl} and dtype={dtype}')


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    my_func2()
