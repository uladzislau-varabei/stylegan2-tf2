# StyleGAN2 - TensorFlow 2.x

![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![TensorFlow 2.9.1](https://img.shields.io/badge/tensorflow-2.9.1-green.svg?style=plastic)
![CUDA Toolkit 11.6.2](https://img.shields.io/badge/cudatoolkit-11.6.2-green.svg?style=plastic)
![cuDNN 8.4.1.50](https://img.shields.io/badge/cudnn-8.4.1.50-green.svg?style=plastic)

TensorFlow 2 implementation of the paper 
**"Analyzing and Improving the Image Quality of StyleGAN"** (https://arxiv.org/abs/1912.04958) <br>
The code is based on the official implementation: https://github.com/NVlabs/stylegan2. 
Some parts have also been taken from the official implementation of the next paper in the StyleGAN series: 
https://github.com/NVlabs/stylegan2-ada.

*Note:* the code is under active development and heavily relies on StyleGAN port. Many things haven't yet been implemented.

This implementation allows finer control of a training process and model complexity via options in configs.


## Training

To train a model one needs to:

1. Define a training config (see `configs` section for details).<br>
   *Note:* paths to images should be saved in a separate *.txt* file, 
   which is to be provided under key `images_paths_filename` in config.
2. Optionally configure gpu memory options (consider **GPU memory usage** section).
3. Optionally set training mode (consider **Training speed** section).
4. Start training with command:

> python train.py --config path_to_config (e.g. --config default_config.json)


## Inference

To run inference consider file `inference.py`. <br>
Example call:

> python .\inference.py --config_path .\configs\lsun_living_room.json  --weights_path .\models\debug_model\weights\step10000\G_model_smoothed.h5 --image_fname images --grid_cols 4 --grid_rows 3


## Configs
Examples of configs are available in `configs` folder.

Paths to images should be saved in a separate *.txt* file, which is to be provided under key `images_paths_filename` in config.

Configs which were used in the official implementation to train on FFHQ dataset:
* `paper_config_ffhq_res1024_full.json` — all values, almost all keys have default values;
* `paper_config_ffhq_res1024_short.json` — similar to the previous config except that omitted keys automatically use default values;
* `paper_config_ffhq_res1024_short_fast.json` — similar to the previous config but with all available speed-ups (mixed precision, XLA, fused bias and activation layer). 

*Note:* options related to summaries are not aligned with the values in the official implementation. Set them according to your needs.

For debugging, it's convenient to use `debug_config.json`.

All possible options and their default values can be found in file `config.py`.


## Training speed

To get maximum performance consider usage of use mixed precision training, which not just speeds operations up 
(especially on Nvidia cards with compute capability 7.0 or higher, e.g., Turing or Ampere GPUs), but also allows to increase batch size.

Some notes about the tricks to enable stable mixed precision training (inspired by one of next papers from the same authors):
* Enable mixed precision only for the N (set to 4 in the official implementation) highest resolutions;
* Clamp the output of every convolutional layer to 2^8, i.e., an order of magnitude wider range than is needed in practise;
* Normalization in modulation/demodulation (still not implemented).

Enabling XLA (Accelerated Linear Algebra, jit compilation) should improve training speed and memory usage.


## GPU memory usage

To control GPU memory usage one can refer to a function `prepare_gpu()` in `tf_utils.py`. 
<br>
Depending on your operating system and use case you might want to change memory managing. 
By default, on Linux all available memory is used, while on Windows memory is limited with some reasonable number to allow use of PC (such as opening browsers with small number of tabs).
<br>
*Note:* the code was used with GPUs with 8 Gb of memory, so if your card has more/less memory it is strongly recommended to consider modifying `prepare_gpu()` function and batch size in training configs. 


## System requirements

* The code was tested on Windows (and will be on Linux later, probably). 
* The following software should be installed on your machine:
```
- NVIDIA driver 516.01 or newer
- TensorFlow-gpu 2.9.1  or newer
- CUDA Toolkit 11.6.2 or newer
- cuDNN 8.4.1.50 or newer
- other dependencies ..
```
* For some reason on Windows 10 with mentioned versions of NVIDIA libraries CUPTI must be manually configured. To do this:
  - Go to folder `c:\Program Files\NVIDIA Corporation\` and search for files `cupti*.dll`. 
  - Copy all of them to your CUPTI folder. 
    Let `cuda_base = c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`. 
    Then it would be `{cuda_base}\extras\CUPTI\lib64\`. 
    File `cupti64_2020.3.1.dll` already was there.
  - Add CUPTI path to `Path` variable: `{cuda_base}\extras\CUPTI\lib64`
  - If you still see error messages try copying all found and existing CUPTI`.dll` files to `{cuda_base}\bin`.
  - Thanks to https://stackoverflow.com/questions/56860180/tensorflow-cuda-cupti-error-cupti-could-not-be-loaded-or-symbol-could-not-be.
  Answer by `Malcolm Swaine`.
    
*Note:* software versions should be consistent, i.e., if you use TensorFlow from *pip* 
you should check CUDA and cuDNN versions on the official TensorFlow site.


## Metrics

Supported metrics are:
* Perceptual Path Length (PPL)
  - Similarly to the official implementation it supports a number of options:
      * Space: *w*, *z*
      * Sampling method: *full*, *end*
      * Epsilon: default is *1e-4*
      * Optional face crop for dataset with human faces (default is *False*) 
      * Number of samples (the official implementation uses 100k, which takes lots of time to run, 
        so consider using a lower value, e.g., 20k or 50k)
  - To calculate the metric when the resolution of generated images is less/greater than 256 (VGG was trained for 224) 
    images are naively upsampled/downsampled to resolution 256, if their resolution is lower/higher than that.
      * Resolution along width dimension is used for comparison. 
        For wide dataset height dimension is scaled according to `dataset_hw_ratio` value. 
        The same scale factor is used for width and height.
      * Probably images should not be upsampled/downsampled. It's not obvious how the case is handled in the official implementation.
  - Supports mixed precision and XLA.
  - Implementation is inspired by *TensorFlow 2* port of lpips model by `moono`: https://github.com/moono/lpips-tf2.x.
* Frechet Inception Distance (FID)
  - To calculate the metric when the resolution of generated images is less/greater than 256 (Inception was trained for 299) 
    images are naively upsampled/downsampled to resolution 256, if their resolution is lower/higher than that.
      * Resolution along width dimension is used for comparison. 
        For wide dataset height dimension is scaled according to `dataset_hw_ratio` value. 
        The same scale factor is used for width and height.
      * Probably images should not be upsampled/downsampled. It's not obvious how the case is handled in the official implementation.
   - Supports mixed precision and XLA.
  

## Further improvements

- Add support for dataset with labels;
- Add multi GPU support;
- Fix training in a single process;
- Fix problems with name scopes inside `tf.function()`. 
  The current solution relies on the answer by `demmerichs`: https://github.com/tensorflow/tensorflow/issues/36464.
  