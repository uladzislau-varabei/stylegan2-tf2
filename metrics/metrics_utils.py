from copy import deepcopy
import logging

from .frechet_inception_distance import FID
from .perceptual_path_length import PPL


BENCHMARK_NUM_SAMPLES = 3000

DEFAULT_FID_PARAMS = {
    'num_samples': 25000,
    'crop_face': False
}

DEFAULT_PPL_PARAMS = {
    'num_samples': 25000,
    'epsilon': 1e-4,
    'space': 'z',
    'sampling': 'full',
    'crop_face': False
}

METRICS_DICT = {
    'FID': (FID, DEFAULT_FID_PARAMS),
    'PPL': (PPL, DEFAULT_PPL_PARAMS)
}


def remove_dataset_info(d):
    new_d = deepcopy(d)
    rm_key = 'dataset_params'
    for k in list(d.keys()):
        if k == rm_key:
            new_d.pop(rm_key)
    return new_d


def setup_metrics(image_size, hw_ratio, dataset_params, use_fp16, use_xla, model_name, metrics, benchmark_mode=False):
    metrics_objects = []
    if metrics is None:
        metrics = {}
    for m_name, m_kwargs in metrics.items():
        assert m_name in METRICS_DICT.keys(), f"Metric '{m_name}' is not implemented, see metrics_dict"
        m_class, m_def_kwargs = METRICS_DICT[m_name]
        if benchmark_mode:
            m_kwargs = deepcopy(m_kwargs)
            m_kwargs['num_samples'] = BENCHMARK_NUM_SAMPLES
        # Note: order of dicts is important to avoid constant use of default params
        metric_kwargs = {
            **{
                'image_size'    : image_size,
                'hw_ratio'      : hw_ratio,
                'dataset_params': dataset_params,
                'use_fp16'      : use_fp16,
                'use_xla'       : use_xla,
                'model_name'    : model_name
            },
            **m_def_kwargs,
            **m_kwargs
        }
        logging.info(f"Metric '{m_name}' uses the following kwargs:")
        logging.info(remove_dataset_info(metric_kwargs))
        metrics_objects.append(
            m_class(**metric_kwargs)
        )
    logging.info('All metrics setup')
    return metrics_objects
