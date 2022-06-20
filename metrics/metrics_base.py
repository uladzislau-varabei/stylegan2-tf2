from config import Config as cfg
from utils import NCHW_FORMAT, validate_data_format
from tf_utils import get_compute_dtype, naive_downsample, naive_upsample


class MetricBase:

    def __init__(self, hw_ratio, num_samples, crop_face,
                 dataset_params, use_fp16, use_xla, model_name, **kwargs):
        self.hw_ratio = hw_ratio
        assert isinstance(num_samples, int)
        self.num_samples = num_samples
        self.crop_face = crop_face
        self.data_format = dataset_params[cfg.DATA_FORMAT]
        validate_data_format(self.data_format)
        self.dataset_params = dataset_params
        self.use_fp16 = use_fp16
        self.compute_dtype = get_compute_dtype(self.use_fp16)
        self.use_xla = use_xla
        self.model_name = model_name

    def maybe_crop_face(self, images):
        # Crop only the face region
        if self.crop_face:
            if self.data_format == NCHW_FORMAT:
                c = int(images.shape[2] // 8)
                images = images[:, :, (c * 3) : (c * 7), (c * 2) : (c * 6)]
            else:  # data_format == NHWC_FORMAT
                c = int(images.shape[2] // 8)
                images = images[:, (c * 3) : (c * 7), (c * 2) : (c * 6), :]
        return images

    def adjust_resolution(self, images, target_size, min_size):
        if self.data_format == NCHW_FORMAT:
            shape_idx = 3
        else: # data_format == NHWC_FORMAT:
            shape_idx = 2
        if images.shape[shape_idx] > target_size:
            factor = images.shape[shape_idx] // target_size
            images = naive_downsample(images, factor, data_format=self.data_format)
        elif images.shape[shape_idx] < min_size:
            factor = min_size // images.shape[shape_idx]
            images = naive_upsample(images, factor, data_format=self.data_format)
        return images

    def scale_dynamic_range_for_imagenet(self, images):
        return (images + 1) * (255 / 2)

    def process_images(self, images):
        raise NotImplementedError

    def get_batch_size(self, input_batch_size):
        # TODO: determine max batch size. For now just always use 32 for small resolutions for Inception and PPL
        if input_batch_size <= 8:
            # Case for high resolutions
            batch_size = 16 # Worked for 512 + transition
        else:
            batch_size = min(max(input_batch_size, 32), 32)
        return batch_size

    def run_metric(self, **kwargs):
        raise NotImplementedError
