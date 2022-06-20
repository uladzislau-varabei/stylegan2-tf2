import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from metrics.metrics_base import MetricBase
from metrics.lpips_models.lpips_tensorflow import vgg_perceptual_metric_model
from utils import NCHW_FORMAT,to_hw_size, clean_array
from tf_utils import toNHWC_AXIS, toNCHW_AXIS, lerp, generate_latents, enable_random_noise, disable_random_noise,\
    enable_mixed_precision_policy, disable_mixed_precision_policy, extract_images


#----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=[-1], keepdims=True))


# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    # Make sure acos inputs have right boundaries (due to numeric rounds)
    d = tf.clip_by_value(d, -1.0, 1.0)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)


#----------------------------------------------------------------------------

class PPL(MetricBase):

    def __init__(self, image_size, hw_ratio, num_samples, epsilon, space, sampling, crop_face,
                 dataset_params, use_fp16, use_xla, model_name, **kwargs):
        super(PPL, self).__init__(
            hw_ratio=hw_ratio, num_samples=num_samples, crop_face=crop_face, dataset_params=dataset_params,
            use_fp16=use_fp16, use_xla=use_xla, model_name=model_name
        )
        assert space in ['w', 'z']
        assert sampling in ['full', 'end']
        self.space = space
        self.sampling = sampling
        self.epsilon = epsilon
        self.norm_constant = 1. / (epsilon ** 2)
        # Required field for each metric class (used in TensorBoard)
        self.name = f'PPL_{space}_{sampling}_{num_samples // 1000}k'

        # VGG was built for 224x224 images, so images are always upscaled/downscaled to size 256 (height is scaled accordingly for wise datasets)
        self.min_size = 256
        self.target_size = 256
        self.image_size = to_hw_size(self.target_size, hw_ratio)

        # TF 2.x port of vgg16_zhang_perceptual
        model_path = os.path.join('metrics', 'lpips_models', 'weights', 'vgg16_perceptual') + '.h5'
        if self.use_fp16:
            enable_mixed_precision_policy()
        # Note: input images should be in NHWC format in range (0, 255)
        self.base_model = vgg_perceptual_metric_model(self.image_size, model_path)
        #  Note: for some reason model = tf.function(lambda x: model(x, args), ...) doesn't work
        self.lpips_model = tf.function(lambda x: self.base_model(x, training=False), jit_compile=self.use_xla)
        if self.use_fp16:
            disable_mixed_precision_policy()

    @tf.function
    def process_images(self, images):
        images = self.maybe_crop_face(images)
        images = extract_images(images, self.hw_ratio, self.data_format)
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        images = self.adjust_resolution(images, self.target_size, self.min_size)
        # Scale dynamic range from [-1,1] to [0,255] for VGG.
        images = self.scale_dynamic_range_for_imagenet(images)
        # Convert images to network format (NHWC).
        if self.data_format == NCHW_FORMAT:
            images = tf.transpose(images, toNHWC_AXIS)
        return images

    def evaluate_distance_for_batch(self, batch_size, G_mapping, G_synthesis):
        # Generate random latents and interpolation t-values.
        # TODO: change to use a noise consistent with the one used by the model.
        lat_t01 = generate_latents(batch_size * 2, G_mapping.input_shape[1],  self.compute_dtype)
        lerp_t = tf.random.uniform([batch_size], 0.0, 1.0 if self.sampling == 'full' else 0.0, dtype=self.compute_dtype)

        # Interpolate in W or Z.
        if self.space == 'w':
            dlat_t01 = G_mapping(lat_t01, training=False)
            dlat_t0, dlat_t1 = dlat_t01[0::2], dlat_t01[1::2]
            dlat_e0 = lerp(dlat_t0, dlat_t1, lerp_t[:, tf.newaxis, tf.newaxis])
            dlat_e1 = lerp(dlat_t0, dlat_t1, lerp_t[:, tf.newaxis, tf.newaxis] + self.epsilon)
            dlat_e01 = tf.reshape(tf.stack([dlat_e0, dlat_e1], axis=1), dlat_t01.shape)
        else:  # space == 'z'
            lat_t0, lat_t1 = lat_t01[0::2], lat_t01[1::2]
            lat_e0 = slerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis])
            lat_e1 = slerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis] + self.epsilon)
            lat_e01 = tf.reshape(tf.stack([lat_e0, lat_e1], axis=1), lat_t01.shape)
            dlat_e01 = G_mapping(lat_e01, training=False)

        # Synthesize images.
        images = G_synthesis(dlat_e01, training=False)
        images = self.process_images(images)

        # Evaluate perceptual distance.
        img_e0, img_e1 = images[0::2], images[1::2]
        # Normalize after rejecting outliers to avoid overflows. Final result doesn't change.
        batch_distance = self.lpips_model([img_e0, img_e1])
        return batch_distance

    def run_metric(self, input_batch_size, G_model):
        G_mapping = G_model.G_mapping
        G_synthesis = G_model.G_synthesis

        randomize_noise = G_model.randomize_noise
        if randomize_noise:
            # This line is very important. Otherwise, images might have visible differences,
            # which leads to very high PPl scores, e.g., 2.5M-3.5M.
            disable_random_noise(G_synthesis)

        # Sampling loop.
        all_distances = []
        batch_size = self.get_batch_size(input_batch_size)
        for _ in tqdm(range(0, self.num_samples, batch_size), desc='PPL metric steps'):
            all_distances.append(self.evaluate_distance_for_batch(batch_size, G_mapping, G_synthesis).numpy())
        all_distances = np.concatenate(all_distances, axis=0)
        all_distances = clean_array(all_distances)

        if randomize_noise:
            enable_random_noise(G_synthesis)

        # Reject outliers.
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)

        # Normalize distance.
        dist = self.norm_constant * np.mean(filtered_distances)
        return dist
