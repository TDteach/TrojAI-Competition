# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import re
import os
from collections import OrderedDict
import torch
import json

def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list(
            dict.fromkeys(
                [
                    re.sub(
                        "\\.(weight|bias|running_(mean|var)|num_batches_tracked)",
                        "",
                        item,
                    )
                    for item in layer_names
                ]
            )
        )
        layer_map = OrderedDict(
            {
                base_layer_name: [
                    layer_name
                    for layer_name in layer_names
                    if re.match(f"{base_layer_name}\.+", layer_name) is not None
                ]
                for base_layer_name in base_layer_names
            }
        )
        model_layer_map[model_class] = layer_map

    return model_layer_map



def wrap_network_prediction(boxes, labels):
    """

    Args:
        boxes: numpy array [N x 4] of the box coordinates
        labels: numpy array [N] of the labels

    Returns:
        list({'bbox': box, 'label': label})
    """
    pred = list()
    for k in range(boxes.shape[0]):
        ann = dict()
        ann['bbox'] = boxes[k, :].tolist()
        ann['label'] = labels[k]
        pred.append(ann)
    return pred


def load_model(model_filepath: str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """
    model = torch.load(model_filepath)
    model_class = model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
    )

    if model_class == 'SSD':
        s = model_repr['head.classification_head.module_list.5.bias'].shape
        pass
    elif model_class == 'DetrForObjectDetection':
        s = model_repr['class_labels_classifier.bias'].shape
        pass
    elif model_class == 'FasterRCNN':
        s = model_repr['roi_heads.box_predictor.cls_score.bias'].shape
        pass
    # print(model_class, n_classes, s[0])

    return model, model_repr, model_class


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:
        Ground truth value (int)
    """

    with open(os.path.join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)


def load_model_info(model_dirpath: str, model_class: str, ground_truth: int):
    with open(os.path.join(model_dirpath, 'config.json'), 'r') as fh:
        jdata = json.load(fh)
    n_classes = int(jdata['py/state']['number_classes'])

    n_triggers = jdata['py/state']['num_triggers']
    if n_triggers is None: n_triggers = 0

    prefix = 'ObjectDetection'
    suffix = 'TriggerExecutor'
    if n_triggers > 0:
        d = jdata['py/state']['triggers'][0]
        z = d['py/object'].split('.')[-1]
        trigger_type = z[len(prefix):-len(suffix)]
    else:
        trigger_type = None


    return {
        'n_classes': n_classes,
        'model_path': os.path.join(model_dirpath, 'model.pt'),
        'model_class': model_class,
        'ground_truth': ground_truth,
        'trigger_type': trigger_type,
    }


def load_models_dirpath(models_dirpath, return_info=False):
    model_repr_dict = {}
    model_ground_truth_dict = {}
    model_info_dict = {}

    nn = []

    for model_path in models_dirpath:

        model, model_repr, model_class = load_model(os.path.join(model_path, "model.pt"))
        model_ground_truth = load_ground_truth(model_path)
        model_info = load_model_info(model_path, model_class, model_ground_truth)

        nn.append(model_info['n_classes'])

        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []
            model_info_dict[model_class] = []

        model_repr_dict[model_class].append(model_repr)
        model_ground_truth_dict[model_class].append(model_ground_truth)
        model_info_dict[model_class].append(model_info)

    print(sorted(set(nn)))

    if return_info:
        return model_repr_dict, model_ground_truth_dict, model_info_dict
    return model_repr_dict, model_ground_truth_dict
