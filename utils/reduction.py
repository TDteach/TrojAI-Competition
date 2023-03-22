import importlib

import numpy as np
from tqdm import tqdm


def feature_reduction(model, weight_table, max_features, max_features_layer):
    outputs = {}
    n_bins = len(weight_table)
    n_layers = len(model)
    sm = sum([l.shape[0] for l in model.values()])
    tf = max_features / sm
    bin_s = [(i+1)/n_bins*sm for i in range(n_bins)]
    cu_i = 0
    ac_s = 0
    id_layer = 0
    a = []
    for (layer, weights) in model.items():
        id_layer += 1
        if id_layer == n_layers:
            out_f = max_features - sum(outputs.values())
        else:
            out_f = 0
            z = weights.shape[0]
            while ac_s + z > bin_s[cu_i]:
                dif = bin_s[cu_i]-ac_s
                out_f += dif * weight_table[cu_i]
                z -= dif
                ac_s += dif
                cu_i += 1
            ac_s += z
            out_f += z * weight_table[cu_i]
            out_f = int(out_f * tf)
        # assert out_f > 0
        a.append(out_f)
        outputs[layer] = out_f

    z = 0
    order = np.argsort(a)
    for i in order[::-1]:
        if a[i] > max_features_layer:
            z += a[i]-max_features_layer
            a[i] = max_features_layer
        else:
            bon = min(z, max_features_layer-a[i])
            a[i] += bon
            z -= bon

    i = 0
    for (layer, weights) in model.items():
        outputs[layer] = a[i]
        i += 1


    return outputs


def init_feature_reduction(output_feats):
    fr_algo = "sklearn.decomposition.FastICA"
    fr_algo_mod = ".".join(fr_algo.split(".")[:-1])
    fr_algo_class = fr_algo.split(".")[-1]
    mod = importlib.import_module(fr_algo_mod)
    fr_class = getattr(mod, fr_algo_class)
    return fr_class(n_components=output_feats)


def init_weight_table(random_seed, mean, std, scaler):
    rnd = np.random.RandomState(seed=random_seed)
    return np.sort(rnd.normal(mean, std, 100)) * scaler


def fit_feature_reduction_algorithm(model_dict, weight_table_params, input_features, max_features_layer=30):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)
    weight_table *= len(weight_table)/np.sum(weight_table)

    for (model_arch, models) in model_dict.items():
        print(model_arch)
        layers_output = feature_reduction(models[0], weight_table, input_features, max_features_layer)

        layer_transform[model_arch] = {}
        for (layers, output) in tqdm(layers_output.items()):
            if output > 0:
                layer_transform[model_arch][layers] = init_feature_reduction(output)
                s = np.stack([model[layers] for model in models])
                layer_transform[model_arch][layers].fit(s)
            else:
                layer_transform[model_arch][layers] = None

    return layer_transform


def use_feature_reduction_algorithm(layer_transform, model):
    out_model = np.array([[]])

    for (layer, weights) in model.items():
        if layer_transform[layer] is None:
            continue
        out_model = np.hstack((out_model, layer_transform[layer].transform([weights])))

    return out_model
