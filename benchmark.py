import argparse

from utils import load_config, BENCHMARK_MODE
from tf_utils import prepare_gpu
from model import StyleGAN2


def parse_args():
    parser = argparse.ArgumentParser(description='Script to benchmark StyleGAN2 model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to benchmark (json format)',
        required=True
    )
    parser.add_argument(
        '--images',
        help='Number of images to run through models',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--run_metrics',
        help='Run metrics in the config? False, if not provided',
        action='store_true'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Example call:
    # python .\benchmark.py --config_path .\configs\lsun_living_room.json --images 1000
    #
    # python .\benchmark.py --config_path .\configs\lsun_living_room.json --images 1000 --run_metrics
    # python .\benchmark.py --config_path .\configs\lsun_car_512x384.json --images 1000 --run_metrics
    args = parse_args()

    images = args.images
    run_metrics = args.run_metrics
    config = load_config(args.config_path)

    # prepare_gpu(mode='growth')
    prepare_gpu()
    StyleGAN2_model = StyleGAN2(config, mode=BENCHMARK_MODE)
    # Note: script benchmarks only model training time, metrics and other post train step actions are not run
    StyleGAN2_model.run_benchmark_stage(images, run_metrics)


########## --- Results --- ##########
# | Res | FP16 res | XLA | Upfirdn IMPL | BS | MBSTD GS | GBASE | DBASE | G loss shrink | G map act | G act | D act | Per layer compile | Fast model | Imgs/persec | Compile | Max mem | Cur mem |
#  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# | 256 |     5    |  +  | custom_grad  | 4  |    2     | 8192  | 8192  |        2      |   lrelu   | lrelu | lrelu |        +          |      -     |     5.661   |         |         |         |
#
#
#
#
#
#
#

