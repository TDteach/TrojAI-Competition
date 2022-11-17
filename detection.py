import os
import sys
import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet as ResNet50
from timm.models.vision_transformer import VisionTransformer as ViT
from torchvision.models.mobilenetv2 import MobileNetV2 as MobileNetV2

from training import ext_features_of_model, MNIST_Detection_Network

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


if __name__ == '__main__':
    pass
