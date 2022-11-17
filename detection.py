import os
import sys
import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet as ResNet50
from timm.models.vision_transformer import VisionTransformer as ViT
from torchvision.models.mobilenetv2 import MobileNetV2 as MobileNetV2

from training import ext_features_of_model, MNIST_Detection_Network

import pickle
import numpy as np

def get_model_type(model, configs):
    model_types = configs['model_types']
    for ty in model_types:
        cls = getattr(sys.modules[__name__], ty)
        if isinstance(model, cls):
            return ty
    return None

def weight_detection(model, parameters_dirpath, configs):
    md_type = get_model_type(model, configs)
    _configs = configs[md_type]
    print(_configs)

    channels_st = _configs['channels_st']
    fet = ext_features_of_model(model, channels_st=channels_st)

    feature_dim = fet.shape[0]
    classifier = MNIST_Detection_Network(in_channels=feature_dim, n_filters=32)
    cls_path = os.path.join(parameters_dirpath, md_type+'.pd')
    classifier.load_state_dict(torch.load(cls_path))
    classifier.eval()
    classifier.cuda()

    fet_tensor = torch.from_numpy(fet)
    fet_tensor = torch.unsqueeze(fet_tensor, dim=0)

    logits = classifier(fet_tensor.float().cuda())
    probs = F.softmax(logits, dim=-1)

    return probs[0, 1].item()


def weight_detection_v2(model, parameters_dirpath, configs):
    md_type = get_model_type(model, configs)
    _configs = configs[md_type]
    print(_configs)

    channels_st = _configs['channels_st']
    fet = ext_features_of_model(model, channels_st=channels_st)

    cls_path = os.path.join(parameters_dirpath, md_type+'.pd')
    with open(cls_path, 'rb') as f:
        cls_model = pickle.load(f)
    best_order = cls_model['best_order']

    all_inputs = fet[best_order, :]
    all_inputs = np.expand_dims(all_inputs, axis=0)

    probs_list = list()
    num_channels = all_inputs.shape[1]
    for ch in range(num_channels):
        fet = all_inputs[:, ch, :]

        clf = cls_model[ch]
        probs = clf.predict_proba(fet)[:, 1]

        probs_list.append(probs)

    pred_scores = np.transpose(np.asarray(probs_list))

    rf_clf = cls_model['rf_clf']
    probs = rf_clf.predict_proba(pred_scores)

    return probs[0,1]


if __name__ == '__main__':
    pass
