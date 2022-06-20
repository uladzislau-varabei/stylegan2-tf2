import os
import logging

import tensorflow as tf

from config import Config as cfg
from utils import NCHW_FORMAT, NHWC_FORMAT, validate_data_format, validate_hw_ratio, to_hw_size
from tf_utils import DEFAULT_DATA_FORMAT, toNCHW_AXIS, toNHWC_AXIS


DEFAULT_DATASET_HW_RATIO             = cfg.DEFAULT_DATASET_HW_RATIO
DEFAULT_DATASET_N_PARALLEL_CALLS     = cfg.DEFAULT_DATASET_N_PARALLEL_CALLS
DEFAULT_DATASET_N_PREFETCHED_BATCHES = cfg.DEFAULT_DATASET_N_PREFETCHED_BATCHES
DEFAULT_SHUFFLE_DATASET              = cfg.DEFAULT_SHUFFLE_DATASET
DEFAULT_MIRROR_AUGMENT               = cfg.DEFAULT_MIRROR_AUGMENT
DEFAULT_USE_FP16                     = cfg.DEFAULT_USE_MIXED_PRECISION
MAX_CACHE_RESOLUTION                 = 7


def adjust_dynamic_range(data, drange_in, drange_out):
    """
    Adjusts ranges of images, e.g. from [0, 1] to [0, 255]
    """
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = (drange_out[0] - drange_in[0]) * scale
        data = data * scale + bias
    return data


def normalize_images(images):
    return 2.0 * images - 1.0


def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    # Always cast to fp32
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_images(images, res, hw_ratio=1, mirror_augment=DEFAULT_MIRROR_AUGMENT, data_format=DEFAULT_DATA_FORMAT, use_fp16=DEFAULT_USE_FP16):
    # 1. Resize images according to the paper implementation (PIL.Image.ANTIALIAS was used)
    images = tf.image.resize(
        images, size=to_hw_size(2 ** res, hw_ratio), method=tf.image.ResizeMethod.LANCZOS3, antialias=True
    )
    # 2. Make sure that dtype and dynamic range didn't change
    images = tf.cast(images, tf.float32)
    images = tf.clip_by_value(images, 0.0, 1.0)
    # Iteration is not allowed, so just take values by idxs
    s = tf.shape(images)
    n, h, w, c = s[0], s[1], s[2], s[3]
    # 3. Optionally apply augmentations
    if mirror_augment:
        mask = tf.random.uniform([n, 1, 1, 1], 0.0, 1.0)
        mask = tf.tile(mask, [1, h, w, c])
        # See source code for tf.image.flip_left_right
        images = tf.where(mask > 0.5, tf.reverse(images, axis=[2]), images)
    # 4. Adjust dynamic range of images
    images = normalize_images(images)
    # 5. Optionally pad images for wide dataset
    if hw_ratio != 1:
        pad_down = (w - h) // 2
        pad_up = w - h - pad_down
        images = tf.pad(images, [[0, 0], [pad_down, pad_up], [0, 0], [0, 0]])
    # 6. Optionally change data format
    if data_format == NCHW_FORMAT:
        images = tf.transpose(images, toNCHW_AXIS)
    # 7. Optionally convert to fp16
    if use_fp16:
        images = tf.cast(images, tf.float16)
    return images


def create_training_dataset(fpaths, res, batch_size,
                            cache=None,
                            hw_ratio=DEFAULT_DATASET_HW_RATIO,
                            mirror_augment=DEFAULT_MIRROR_AUGMENT,
                            shuffle_dataset=DEFAULT_SHUFFLE_DATASET,
                            data_format=DEFAULT_DATA_FORMAT,
                            use_fp16=DEFAULT_USE_FP16,
                            n_parallel_calls=DEFAULT_DATASET_N_PARALLEL_CALLS,
                            n_prefetched_batches=DEFAULT_DATASET_N_PREFETCHED_BATCHES):
    def get_value(n):
        value = tf.data.AUTOTUNE
        if isinstance(n, int):
            if n > 0:
                value = n
        return value

    validate_data_format(data_format)
    validate_hw_ratio(hw_ratio)

    ds = tf.data.Dataset.from_tensor_slices(fpaths)

    if shuffle_dataset:
        shuffle_buffer_size = len(fpaths)
        logging.info('Shuffling dataset...')
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Split loading and preprocessing to make data pipeline more efficient
    ds = ds.map(lambda x: load_image(x), num_parallel_calls=get_value(n_parallel_calls))

    # cache can be a path to folder where files should be created
    # Note: when working with high resolutions there is no need to cache ds
    # as it consumes too much space on data storage (up to several GBs)
    if res <= MAX_CACHE_RESOLUTION:
        if isinstance(cache, str):
            ds = ds.cache(os.path.join(cache, str(res)))

    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    # Perform vectorized operations
    ds = ds.map(
        lambda x: preprocess_images(
            x, res=res, hw_ratio=hw_ratio, mirror_augment=mirror_augment, data_format=data_format, use_fp16=use_fp16
        ),
        num_parallel_calls=get_value(n_parallel_calls)
    )

    # Fetch batches in the background while model is training
    # If applied after ds.batch() then buffer_size is given in batches,
    # so total number of prefetched elements is batch_size * buffer_size
    ds = ds.prefetch(buffer_size=get_value(n_prefetched_batches))
    return ds
