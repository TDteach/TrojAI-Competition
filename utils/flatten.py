import logging
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


def flatten_layer(model, layer_map):
    nbt_layer = None
    output = None
    for layer in layer_map:
        if "num_batches_tracked" in layer:
            nbt_layer = layer
            continue

        if len(model[layer].shape) == 1:
            model_layer = np.array([model[layer]]).T
        else:
            feats = model[layer].shape[0]
            flat_layer = model[layer].flatten()
            model_layer = flat_layer.reshape(feats, int(flat_layer.shape[0] / feats))
        if output is None:
            output = model_layer
        else:
            output = np.hstack((output, model_layer))
    output = output.flatten()
    if nbt_layer:
        output = np.hstack((output, model[nbt_layer]))
    return output


def regularize_layer_parameters(model, layer_map):
    pmean = np.mean(model[layer_map[0]][:])
    pstd = np.std(model[layer_map[0]][:])

    output = None
    for layer in layer_map:
        param = model[layer]
        if len(param.shape) < 4:
            param = np.expand_dims(param, axis=[i for i in range(len(param.shape), 4)])
        shape = param.shape
        assert len(shape) == 4

        param = np.reshape(param, (shape[0]*shape[1], shape[2]*shape[3]))
        param = param.T

        if output is None:
            output = param
        else:
            if param.shape[1] < output.shape[1]:
                ratio = output.shape[1]//param.shape[1]
                param = np.expand_dims(param, axis=2)
                param = np.tile(param, ratio)
                param = np.reshape(param, (param.shape[0], param.shape[1]*param.shape[2]))
                assert param.shape[1] == output.shape[1]
            output = np.vstack((output, param))

    output = (output-pmean)/pstd
    return output



def regularize_model_parameters(input_model, model_layers):
    new_model = OrderedDict()
    for (layer, layer_map) in model_layers.items():
        if len(layer_map) > 2: continue
        if len(layer_map) > 0:
            new_model[layer] = regularize_layer_parameters(input_model, layer_map)
        else:
            new_model[layer] = regularize_layer_parameters(input_model, [layer])
    return new_model



def flatten_model(input_model, model_layers):
    new_model = OrderedDict()
    for (layer, layer_map) in model_layers.items():
        new_model[layer] = (
            flatten_layer(input_model, layer_map)
            if len(layer_map) > 0
            else input_model[layer].flatten()
        )
        assert len(new_model[layer].shape) == 1
    return new_model


def flatten_models(model_repr_dict, model_layer_map):
    """Flatten a list of models

    Args:
        model_repr_dict:
        model_layer_map:

    Returns:
    """
    flat_models = {}

    for _ in range(len(model_repr_dict)):
        (model_arch, models) = model_repr_dict.popitem()
        if model_arch not in flat_models.keys():
            flat_models[model_arch] = []

        logging.info("Flattenning %s models...", model_arch)
        for _ in tqdm(range(len(models))):
            model = models.pop(0)
            flat_models[model_arch].append(
                # flatten_model(model, model_layer_map[model_arch])
                regularize_model_parameters(model, model_layer_map[model_arch])
            )

    return flat_models
