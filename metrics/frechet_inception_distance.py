import os
from tqdm import tqdm

import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_input

from metrics.metrics_base import MetricBase
from dataloader_utils import create_training_dataset
from utils import CACHE_DIR, MODELS_DIR, to_hw_size
from tf_utils import toNHWC_AXIS, NCHW_FORMAT,\
    enable_mixed_precision_policy, disable_mixed_precision_policy, extract_images


FID_DIR = 'fid'
MU_REAL_KEY = 'mu_real'
SIGMA_REAL_KEY = 'sigma_real'


class FID(MetricBase):

    def __init__(self, image_size, hw_ratio, num_samples, crop_face, dataset_params, use_fp16, use_xla, model_name, **kwargs):
        super(FID, self).__init__(
            hw_ratio=hw_ratio, num_samples=num_samples, crop_face=crop_face, dataset_params=dataset_params,
            use_fp16=use_fp16, use_xla=use_xla, model_name=model_name
        )
        # Required field for each metric class (used in TensorBoard)
        self.name = f'FID_{num_samples // 1000}k'
        self.cache_dir = os.path.join(MODELS_DIR, model_name, CACHE_DIR)
        self.cache_file = os.path.join(self.cache_dir, self.name + f'_size{image_size}.npz')

        # Inception was built for 299x299 images, so images are always upscaled/downscaled to size 256 (height is scaled accordingly for wise datasets)
        self.min_size = 256
        self.target_size = 256
        self.image_size = to_hw_size(self.target_size, hw_ratio)
        self.image_res_log2 = int(np.log2(self.image_size[1]))

        if self.use_fp16:
            enable_mixed_precision_policy()
        # With pooling specified output shape will always be (None, 2048), so no need to provide input shape
        self.base_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        self.activations_shape = [self.num_samples, self.base_model.output_shape[1]]
        self.activations_dtype = np.float16 if self.use_fp16 else np.float32
        # Note: for some reason model = tf.function(lambda x: model(x, args), ...) doesn't work
        self.inception_model = tf.function(lambda x: self.base_model(x, training=False), jit_compile=self.use_xla)
        if self.use_fp16:
            disable_mixed_precision_policy()

    def create_images_dataset(self, batch_size):
        self.dataset_params.update(res=self.image_res_log2, batch_size=batch_size, cache=False)
        return create_training_dataset(**self.dataset_params)

    @tf.function
    def process_images(self, images):
        images = self.maybe_crop_face(images)
        images = extract_images(images, self.hw_ratio, self.data_format)
        # Upsample image to 256x256 if it's smaller than that. Inception was built for 299x299 images.
        images = self.adjust_resolution(images, self.target_size, self.min_size)
        # Scale dynamic range from [-1,1] to [0,255] for Inception.
        images = self.scale_dynamic_range_for_imagenet(images)
        # Convert images to network format (NHWC).
        if self.data_format == NCHW_FORMAT:
            images = tf.transpose(images, toNHWC_AXIS)
        # Prepare images for Inception model.
        images = preprocess_inception_input(images, data_format='channels_last')
        return images

    def compute_activations_stats(self, activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate_on_reals(self, batch_size):
        if os.path.exists(self.cache_file):
            data = np.load(self.cache_file)
            mu_real, sigma_real = data[MU_REAL_KEY], data[SIGMA_REAL_KEY]
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            images_dataset = iter(self.create_images_dataset(batch_size))
            activations = np.empty(self.activations_shape, dtype=self.activations_dtype)
            for idx in tqdm(range(0, self.num_samples, batch_size), 'FID metric reals steps'):
                start = idx * batch_size
                end = min(start + batch_size, self.num_samples)
                real_images = next(images_dataset)
                real_images = self.process_images(real_images)
                activations[start:end] = self.inception_model(real_images).numpy()[:(end-start)]
            mu_real, sigma_real = self.compute_activations_stats(activations)
            np.savez(self.cache_file, **{MU_REAL_KEY: mu_real, SIGMA_REAL_KEY: sigma_real})
        return mu_real, sigma_real

    def evaluate_on_fakes(self, batch_size, G_model):
        activations = np.empty(self.activations_shape, dtype=self.activations_dtype)
        for idx in tqdm(range(0, self.num_samples, batch_size), 'FID metric fakes steps'):
            start = idx * batch_size
            end = min(start + batch_size, self.num_samples)
            latents = G_model.generate_latents(batch_size)
            fake_images = G_model(latents, training=False, validation=True)
            fake_images = self.process_images(fake_images)
            activations[start:end] = self.inception_model(fake_images).numpy()[:(end-start)]
        mu_fake, sigma_fake = self.compute_activations_stats(activations)
        return mu_fake, sigma_fake

    def calculate_fid(self, real_stats, fake_stats):
        mu_real, sigma_real = real_stats
        mu_fake, sigma_fake = fake_stats
        m = np.square(mu_fake - mu_real).sum()
        s, _ = sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2.0 * s.real)
        return dist

    def run_metric(self, input_batch_size, G_model):
        batch_size = self.get_batch_size(input_batch_size)
        mu_real, sigma_real = self.evaluate_on_reals(batch_size)
        mu_fake, sigma_fake = self.evaluate_on_fakes(batch_size, G_model)
        try:
            dist = self.calculate_fid((mu_real, sigma_real), (mu_fake, sigma_fake))
        except:
            # Case when for some reason memory can't be allocated
            dist = 0.0
        return dist
