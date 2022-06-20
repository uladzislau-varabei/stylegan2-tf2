import os
import argparse
import time

import numpy as np
import tensorflow as tf

from config import Config as cfg
from checkpoint_utils import load_weights
from utils import INFERENCE_MODE, WEIGHTS_DIR, fast_save_grid, load_config
from tf_utils import DEFAULT_DATA_FORMAT, prepare_gpu, generate_latents, convert_outputs_to_images, run_model_on_batches
from model import StyleGAN2


def parse_args():
    parser = argparse.ArgumentParser(description='Script to run inference for StyleGAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        default=os.path.join('configs', 'lsun_living_room.json'),
        required=True
    )
    parser.add_argument(
        '--weights_path',
        help='Path to a model weights',
        required=True
    )
    # Network options
    parser.add_argument(
        '--truncation_psi',
        help='Style strength multiplier for the truncation trick, if not provided, value from config is used',
        default=None
    )
    parser.add_argument(
        '--truncation_cutoff',
        help='Number of layers for which to apply the truncation trick, if not provided, value from config is used',
        default=None
    )
    parser.add_argument(
        '--disable_truncation',
        help='Save generated image in jpg? If not provided, png will be used',
        action='store_true'
    )
    # Image grid options
    parser.add_argument(
        '--image_fname',
        help='Filename for generated image',
        required=True
    )
    parser.add_argument(
        '--grid_cols',
        help='Number of columns in image grid',
        type=int,
        default=cfg.DEFAULT_VALID_GRID_NCOLS
    )
    parser.add_argument(
        '--grid_rows',
        help='Number of rows in image grid',
        type=int,
        default=cfg.DEFAULT_VALID_GRID_NROWS
    )
    parser.add_argument(
        '--save_in_jpg',
        help='Save generated image in jpg? If not provided, png will be used',
        action='store_true'
    )
    args = parser.parse_args()
    return args


def generate_images(model: tf.keras.Model, truncation_psi: float, truncation_cutoff: int, config: dict):
    start_time = time.time()

    latent_size = config.get(cfg.LATENT_SIZE, cfg.DEFAULT_LATENT_SIZE)
    data_format = config.get(cfg.DATA_FORMAT, DEFAULT_DATA_FORMAT)
    hw_ratio = config.get(cfg.DATASET_HW_RATIO, cfg.DEFAULT_DATASET_HW_RATIO)

    latents = generate_latents(grid_cols * grid_rows, latent_size)
    model_kwargs = {'training': False, 'truncation_psi': truncation_psi, 'truncation_cutoff': truncation_cutoff}
    batch_size = 16
    images = run_model_on_batches(model, model_kwargs, latents, batch_size)
    images = convert_outputs_to_images(images, 2 ** res, hw_ratio=hw_ratio, data_format=data_format).numpy()

    total_time = time.time() - start_time
    print(f'Generated images in {total_time:.3f}s')

    return images


def extract_res_and_stage(p):
    s1 = p.split(WEIGHTS_DIR)[1]
    splits = s1.split(os.path.sep)
    res = int(np.log2(int(splits[2].split('x')[0])))
    stage = splits[3]
    return res, stage


def get_truncation_values(args, config):
    disable_truncation = args.disable_truncation
    if disable_truncation:
        truncation_psi = None
        truncation_cutoff = None
    else:
        truncation_psi = args.truncation_psi
        if truncation_psi is None:
            truncation_psi = config.get(cfg.TRUNCATION_PSI, cfg.DEFAULT_TRUNCATION_PSI)
        truncation_cutoff = args.truncation_cutoff
        if truncation_cutoff is None:
            truncation_cutoff = config.get(cfg.TRUNCATION_CUTOFF, cfg.DEFAULT_TRUNCATION_CUTOFF)
    return truncation_psi, truncation_cutoff


if __name__ == '__main__':
    """
    ----- Example calls -----
    
    1) Disable truncation trick and save output image in jpg
    python .\inference.py --config_path .\configs\lsun_living_room.json --weights_path .\weights\lsun_living_room\256x256\stabilization\step3000000\G_model_smoothed.h5 --disable_truncation --image_fname temp_images --grid_cols 12 --grid_rows 9 --save_in_jpg
    
    2) Provide options values for truncation trick (skip to use default ones from config) and save output image in png
    python .\inference.py --config_path .\configs\lsun_living_room.json  --weights_path .\weights\lsun_living_room\256x256\stabilization\step3000000\G_model_smoothed.h5 --truncation_psi 0.7 --truncation_cutoff 8  --image_fname temp_images --grid_cols 4 --grid_rows 3    
    """
    args = parse_args()

    config = load_config(args.config_path)
    res, stage = extract_res_and_stage(args.weights_path)
    weights_path = args.weights_path
    truncation_psi, truncation_cutoff = get_truncation_values(args, config)

    # Grid image options
    image_fname = args.image_fname
    grid_cols = args.grid_cols
    grid_rows = args.grid_rows
    save_in_jpg = args.save_in_jpg

    prepare_gpu(mode='growth')
    StyleGAN2_model = StyleGAN2(config, mode=INFERENCE_MODE)
    Gs_model = StyleGAN2_model.Gs_object.create_G_model()
    Gs_model = load_weights(Gs_model, weights_path, optimizer_call=False)
    images = generate_images(Gs_model, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, config=config)

    out_dir = 'results'
    fast_save_grid(
        out_dir=out_dir,
        fname=image_fname,
        images=images,
        title=None,
        nrows=grid_rows,
        ncols=grid_cols,
        padding=2,
        save_in_jpg=save_in_jpg
    )
