import os
import sys
import json
import logging
import time

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

from config import Config as cfg


# Recommended for Tensorflow
NCHW_FORMAT = 'NCHW'
# Recommended by Nvidia
NHWC_FORMAT = 'NHWC'

SKIP_ARCHITECTURE   = 'skip'
RESNET_ARCHITECTURE = 'resnet'
SMOOTH_POSTFIX      = '_smoothed'
OPTIMIZER_POSTFIX   = '_optimizer'
RGB_NAME            = 'RGB'
LOD_NAME            = 'lod'
GENERATOR_NAME      = 'G_model'
DISCRIMINATOR_NAME  = 'D_model'

# Dirs
MODELS_DIR         = 'models'
WEIGHTS_DIR        = 'weights'
LOGS_DIR           = 'logs'
TF_LOGS_DIR        = 'tf_logs'
IMAGES_DIR         = 'images'
DATASET_CACHE_DIR  = 'tf_ds_cache'
CACHE_DIR          = 'cache'
EXAMPLE_IMAGES_DIR = 'example_images' # For jupyter notebooks

TRAIN_MODE = 'training'
INFERENCE_MODE = 'inference'
BENCHMARK_MODE = 'benchmark'
DEFAULT_MODE = INFERENCE_MODE

DEBUG_MODE = 'debug_mode'
DEFAULT_DEBUG_MODE = '0'


def should_log_debug_info():
    return int(os.environ.get(DEBUG_MODE, DEFAULT_DEBUG_MODE)) > 0


#----------------------------------------------------------------------------
# Utils.

def prepare_logger(model_name):
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    model_dir = os.path.join(MODELS_DIR, model_name, LOGS_DIR)
    filename = os.path.join(model_dir, f'logs_{model_name}.txt')
    os.makedirs(model_dir, exist_ok=True)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)

    print('Logging initialized!')


def sleep(s):
    print(f"Sleeping {s}s...")
    time.sleep(s)
    print("Sleeping finished")


def format_time(seconds):
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def clean_array(arr):
    nans = np.isnan(arr)
    n_vals = len(arr)
    n_nans = np.count_nonzero(nans)
    message = f'Number of nans in array is {n_nans} or {(100 * (n_nans / n_vals)):.2f}% '
    logging.info(message)
    return arr[~nans]


def load_config(config_path) -> dict:
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def to_int_dict(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def validate_data_format(data_format: str):
    assert data_format in [NCHW_FORMAT, NHWC_FORMAT]


def validate_hw_ratio(hw_ratio):
    assert hw_ratio == 1 or 0.1 < hw_ratio < 1.0


def to_hw_size(image_size, hw_ratio) -> tuple:
    validate_hw_ratio(hw_ratio)
    return (int(hw_ratio * image_size), image_size)


def create_images_dir_path(model_name) -> str:
    return os.path.join(MODELS_DIR, model_name, IMAGES_DIR)


def create_images_grid_title(step) -> str:
    return f'step={step}'


def load_images_paths(config) -> list:
    images_paths_filename = config[cfg.IMAGES_PATHS_FILENAME]
    with open(images_paths_filename, 'r') as f:
        file_lines = f.readlines()
    images_paths = [x.strip() for x in file_lines]

    dataset_n_max_images = int(1000 * config.get(cfg.DATASET_N_MAX_KIMAGES, cfg.DEFAULT_DATASET_N_MAX_KIMAGES))
    if dataset_n_max_images > 0:
        logging.info(f'Dataset number of images: {len(images_paths)}, max number of images: {dataset_n_max_images}')
        if len(images_paths) > dataset_n_max_images:
            logging.info(f'Reduced dataset to {dataset_n_max_images} images')
            images_paths = images_paths[:dataset_n_max_images]

    logging.info(f'Total number of images: {len(images_paths)}')
    return images_paths


def is_last_step(step, n_steps):
    return step == (n_steps - 1)


def should_write_summary(summary_every: int, n_images: int, batch_size: int):
    return (n_images // summary_every > 0 and n_images % summary_every < batch_size) or n_images == batch_size


def level_of_details(res, resolution_log2):
    return resolution_log2 - res + 1


def mult_by_zero(weights):
    return [0. * w for w in weights]


def get_start_fp16_resolution(num_fp16_resolutions, start_resolution_log2, target_resolution_log2):
    # 1) 2 - 4 - start block resolution
    # 2) 3 - 8 - default start resolution
    # 3) 4 - 16
    # 4) 5 - 32
    # 5) 6 - 64
    # 6) 7 - 128
    # 7) 8 - 256
    # 8) 9 - 512
    # 9) 10 - 1024
    if num_fp16_resolutions == 'auto':
        """
        # 1st value: a new init value, 2nd value: taken from the official implementation (N = 4)
        return max(
            min(start_resolution_log2 + 2, target_resolution_log2 - 4 + 1), start_resolution_log2
        )
        """
        # Value from the official implementation (N = 4) of one of the next papers
        return target_resolution_log2 - 4 + 1
    else:
        return target_resolution_log2 - num_fp16_resolutions + 1


def should_use_fp16(res, start_fp16_resolution_log2, use_mixed_precision):
    return res >= start_fp16_resolution_log2 and use_mixed_precision


def adjust_clamp(clamp, use_fp16):
    # If layer doesn't use fp16 then values shouldn't be clamped
    return clamp if use_fp16 is True else None


#----------------------------------------------------------------------------
# Image utils.

def show_collage(imgs, rows, cols, scores):
    fix, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5), dpi=150)
    n = len(imgs)
    i = 0
    add_title_score = True if scores is not None else False
    for r in range(rows):
        for c in range(cols):
            ax[r, c].imshow(imgs[i])
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if add_title_score:
                title = "Image %i, score=%.3f" % (i, scores[i])
            else:
                title = "Image %i" % i
            ax[r, c].set_title(title, fontsize=14)
            i += 1
            if i == n:
                break
        if i == n:
            break
    plt.show()


def save_grid(fname, images, nrows, ncols, title, img_titles=None):
    """
    Should only be used when there are titles for each image, as it's very slow.
    Otherwise use 'fast_save_grid'.
    Note: '.jpg'
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), dpi=150)
    add_img_title = img_titles is not None
    for idx, image in enumerate(images):
        row = idx // nrows
        col = idx % ncols
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
        if add_img_title:
            axes[row, col].set_title(str(img_titles[idx]), fontsize=15)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(wspace=.05, hspace=.2)
    plt.savefig(fname + '.jpg', bbox_inches='tight')
    plt.close()


def fast_make_grid(images, nrows, ncols, padding):
    _, h, w, _ = images.shape
    grid_h = nrows * h + (nrows - 1) * padding
    grid_w = ncols * w + (ncols - 1) * padding

    image_grid = np.zeros((grid_h, grid_w, 3), dtype=images.dtype)
    hp = h + padding
    wp = w + padding

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            image_grid[hp * r: hp * (r + 1) - padding, wp * c: wp * (c + 1) - padding, :] = images[i]
            i += 1

    return image_grid


def plt_save_grid(fname, images, nrows, ncols, padding, title):
    """
    Should only be used for benchmarks.
    If you have title for each image then use 'save_grid' otherwise 'fast_save_grid'
    """
    img_grid = fast_make_grid(images, nrows=nrows, ncols=ncols, padding=padding)
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    ax.imshow(img_grid)
    ax.axis("off")
    ax.set_title(title, fontsize=15)
    plt.savefig(fname + '.jpg', bbox_inches='tight', quality=95)
    plt.close()


def add_title_background(img_array):
    h, w, _ = img_array.shape
    background = np.zeros([int(0.05 * h), w, 3], dtype=img_array.dtype)
    return np.vstack([background, img_array])


def convert_to_pil_image(images):
    """
    :param images: numpy array of dtype=uint8 in range [0, 255]
    :return: PIL image
    """
    return Image.fromarray(images)


def convert_to_pil_image_with_title(img_array, title):
    h, w, _ = img_array.shape
    img_array = add_title_background(img_array)
    img = convert_to_pil_image(img_array)

    # See function add_title_background
    font_size = int(0.025 * h)
    # Font can be stored in a folder with script
    font = ImageFont.truetype("arial.ttf", font_size)

    d = ImageDraw.Draw(img)
    # text_w, text_h = d.textsize(title)
    text_w_start_pos = (w - font.getsize(title)[0]) / 2
    d.text((text_w_start_pos, 0.01 * h), title, fill="white", font=font)
    return img


def fast_save_grid(out_dir, fname, images, nrows, ncols, padding, title, save_in_jpg=False):
    img_grid = fast_make_grid(images, nrows=nrows, ncols=ncols, padding=padding)
    if title is not None:
        img = convert_to_pil_image_with_title(img_grid, title)
    else:
        img = convert_to_pil_image(img_grid)

    os.makedirs(out_dir, exist_ok=True)

    if save_in_jpg:
        img.save(os.path.join(out_dir, fname + '.jpg'), 'JPEG', subsample=0, quality=95, optimize=True)
    else:
        img.save(os.path.join(out_dir, fname + '.png'), 'PNG', optimize=True)
