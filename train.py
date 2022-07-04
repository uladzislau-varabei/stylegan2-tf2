import os
import argparse
import logging
import time
from multiprocessing import Process

# Note: do not import tensorflow here or you won't be able to train each stage in a new process

from config import Config as cfg
from utils import TRAIN_MODE, INFERENCE_MODE, DEBUG_MODE
from utils import load_config, load_images_paths, format_time, prepare_logger, sleep
from tf_utils import prepare_gpu
from model import StyleGAN2


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train StyleGAN2 model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        #default=os.path.join('configs', 'demo_config.json'),
        default=os.path.join('configs', 'debug_config.json'),
        #default=os.path.join('configs', 'lsun_living_room.json'),
        #default=os.path.join('configs', 'lsun_car_512x384.json'),
        #required=True
    )
    args = parser.parse_args()
    return args


def run_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()


def trace_graphs(config):
    prepare_logger(config[cfg.MODEL_NAME])
    pid = os.getpid()
    logging.info(f'Tracing graphs uses PID={pid}')
    prepare_gpu()
    StyleGAN2_model = StyleGAN2(config, mode=INFERENCE_MODE)
    StyleGAN2_model.trace_graphs()


def run_train_stage(config, images_paths):
    # prepare_logger(config[cfg.MODEL_NAME])
    pid = os.getpid()
    logging.info(f'Training of StyleGAN2 model uses PID={pid}')
    # prepare_gpu('limit', memory_limit=5000)
    StyleGAN2_model = StyleGAN2(config, mode=TRAIN_MODE, images_paths=images_paths)
    StyleGAN2_model.run_training()


def train_model(config):
    images_paths = load_images_paths(config)

    """
    run_process(target=trace_graphs, args=(config, ))
    sleep(3)
    print('Manual exit')
    exit()
    """

    train_start_time = time.time()
    logging.info(f'Training StyleGAN2 model...')
    # Training
    """
    # Some problems with training in a separate process
    run_process(
        target=run_train_stage,
        args=(config, images_paths)
    )
    """
    # prepare_gpu('growth')
    # prepare_gpu('limit', memory_limit=5000)
    prepare_gpu('limit')
    run_train_stage(config, images_paths)
    logging.info(f'----------------------------------------------------------------------')
    logging.info('')

    train_total_time = time.time() - train_start_time
    logging.info(f'Training finished in {format_time(train_total_time)}!')


if __name__ == '__main__':
    args = parse_args()

    config = load_config(args.config_path)
    prepare_logger(config[cfg.MODEL_NAME])
    logging.info('Training with the following config:')
    logging.info(config)

    # Should log debug information?
    debug_mode = False
    os.environ[DEBUG_MODE] = '1' if debug_mode else '0'

    train_model(config)
