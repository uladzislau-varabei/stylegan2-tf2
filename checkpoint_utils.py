import os
import json
import logging
import glob
import shutil

from tensorflow.keras import mixed_precision
import h5py

from config import Config as cfg
from utils import should_log_debug_info, MODELS_DIR, WEIGHTS_DIR, OPTIMIZER_POSTFIX


H5_WEIGHTS_KEY = 'weights'
LOSS_SCALE_KEY = 'loss_scale'
STEP_DIR_PREFIX = 'step'
DEFAULT_STORAGE_PATH = cfg.DEFAULT_STORAGE_PATH


#----------------------------------------------------------------------------
# Model saving/loading/remove utils.


def weights_to_dict(model, optimizer_call: bool = False):
    vars = model.trainable_variables if not optimizer_call else model.weights
    if should_log_debug_info():
        print('\nSaving weights:')
        for idx, var in enumerate(vars):
            print(f'{idx}: {var.name}')
    return {var.name: var.numpy() for var in vars}


def load_model_weights_from_dict(model, weights_dict):
    log_debug_info = should_log_debug_info()
    print('\nLoading weights from dict')
    print('\nModel train vars:', model.trainable_variables)
    print('\nDict vars:', list(weights_dict.keys()))
    for var in model.trainable_variables:
        if var.name in weights_dict.keys():
            if log_debug_info:
                print(f'Loading {var.name}')
            var.assign(weights_dict[var.name])

    return model


def save_weights(weights_dict, filename: str):
    f = h5py.File(filename, 'w')
    g = f.create_group(H5_WEIGHTS_KEY)

    for idx, var_name in enumerate(weights_dict.keys()):
        value = weights_dict[var_name]
        shape = value.shape
        dset = g.create_dataset(name=var_name, shape=shape, dtype=value.dtype.name)
        if not shape:
            # Scalar
            dset[()] = value
        else:
            dset[:] = value

    f.flush()
    f.close()


def load_weights_into_dict(var_names, filename: str):
    f = h5py.File(filename, 'r')
    g = f[H5_WEIGHTS_KEY]

    var_dict = {}
    for var_name in var_names:
        if var_name in g:
            # Check for scalar
            try:
                var_dict[var_name] = g[var_name][:]
            except:
                var_dict[var_name] = g[var_name][()]

    f.close()
    return var_dict


def load_weights(model, filename, optimizer_call: bool = False):
    vars = model.trainable_variables if not optimizer_call else model.weights
    var_names = [var.name for var in vars]
    var_dict = load_weights_into_dict(var_names, filename)

    log_debug_info = should_log_debug_info()
    loaded_vars = []
    # Note: another approach is to use set_weights (load ur use existing value) for models and optimizers (maybe with deepcopy?)
    for var in vars:
        if var.name in var_dict.keys():
            if log_debug_info:
                print(f'Loading {var.name}')
            # Might be a Strange line for optimizer
            var.assign(var_dict[var.name])
            loaded_vars.append(var.name)

    # TODO: for debugging, remove later
    # print('\nOptimizer call:', optimizer_call)
    # print('Loaded these vars:\n', loaded_vars)

    return model


def create_model_dir_path(model_name, step=None, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    model_name - name of configuration model
    step - number of all processed images for given resolution and stage
    storage_path - optional prefix path
    """
    step_dir = STEP_DIR_PREFIX + str(step) if step is not None else ''
    model_dir_path = os.path.join(MODELS_DIR, model_name, WEIGHTS_DIR, step_dir)

    if storage_path is not None:
        model_dir_path = os.path.join(storage_path, model_dir_path)

    return model_dir_path


def save_model(model, model_name, model_type,
               step, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    model - a model to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    step - number of all processed images for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    model_dir_path = create_model_dir_path(
        model_name=model_name,
        step=step,
        storage_path=storage_path
    )
    optimizer_call = OPTIMIZER_POSTFIX in model_type
    if optimizer_call and (step is None):
        model_dir_path = os.path.join(model_dir_path, OPTIMIZER_POSTFIX)
    os.makedirs(model_dir_path, exist_ok=True)

    filepath = os.path.join(model_dir_path, model_type + '.h5')
    weights_dict = weights_to_dict(model, optimizer_call=optimizer_call)

    save_weights(weights_dict, filepath)


def save_optimizer_loss_scale(optimizer: mixed_precision.LossScaleOptimizer,
                              model_name: str, model_type: str,
                              step: int, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    optimizer - an optimizer model from which loss scale is to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    step - number of all processed images for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    model_dir_path = create_model_dir_path(
        model_name=model_name,
        step=step,
        storage_path=storage_path
    )
    if step is None:
        model_dir_path = os.path.join(model_dir_path, OPTIMIZER_POSTFIX)
    os.makedirs(model_dir_path, exist_ok=True)

    # This function is only called when loss scale is dynamic
    loss_scale = float(optimizer._loss_scale().numpy())
    if should_log_debug_info():
        print(f'Saved loss scale for {model_type}: {loss_scale}')

    filepath = os.path.join(model_dir_path, model_type + '.json')
    with open(filepath, 'w') as fp:
        json.dump({LOSS_SCALE_KEY: loss_scale}, fp)


def load_model(model, model_name, model_type,
               step, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    model - a model to be loaded
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME], used as a separate dir level
    step - number of all processed images for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of loading model
    """
    model_dir_path = create_model_dir_path(
        model_name=model_name,
        step=step,
        storage_path=storage_path
    )
    optimizer_call = OPTIMIZER_POSTFIX in model_type
    if optimizer_call and (step is None):
        model_dir_path = os.path.join(model_dir_path, OPTIMIZER_POSTFIX)
    assert os.path.exists(model_dir_path),\
        f"Can't load weights: directory {model_dir_path} does not exist"

    filepath = os.path.join(model_dir_path, model_type + '.h5')
    model = load_weights(model, filepath, optimizer_call=optimizer_call)
    return model


def load_optimizer_loss_scale(model_name: str, model_type: str,
                              step: int, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    optimizer - an optimizer model from which loss scale is to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    step - number of all processed images for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    model_dir_path = create_model_dir_path(
        model_name=model_name,
        step=step,
        storage_path=storage_path
    )
    if step is None:
        model_dir_path = os.path.join(model_dir_path, OPTIMIZER_POSTFIX)

    filepath = os.path.join(model_dir_path, model_type +  OPTIMIZER_POSTFIX + '.json')
    with open(filepath, 'r') as fp:
        loss_scale = json.load(fp)[LOSS_SCALE_KEY]

    if should_log_debug_info():
        print(f'Loaded loss scale for {model_type}: {loss_scale}')

    return loss_scale


def remove_old_models(model_name, max_models_to_keep: int, storage_path: str = DEFAULT_STORAGE_PATH):
    """
    model_name - name of configuration model
    max_models_to_keep - max number of models to keep
    storage_path - optional prefix path
    """
    log_debug_info = should_log_debug_info()
    if log_debug_info:
        logging.info('\nRemoving weights...')
    # step and model_type are not used, so just use valid values
    weights_path = create_model_dir_path(
        model_name=model_name,
        step=1,
        storage_path=storage_path
    )
    res_stage_path = os.path.split(weights_path)[0]
    sorted_steps_paths = sorted(
        [x for x in glob.glob(res_stage_path + os.sep + '*') if STEP_DIR_PREFIX in x],
        key=lambda x: int(x.split(STEP_DIR_PREFIX)[1])
    )
    # Remove weights for all steps except the last ones
    for p in sorted_steps_paths[:-max_models_to_keep]:
        shutil.rmtree(p)
        if log_debug_info:
            logging.info(f'Removed weights for path={p}')
