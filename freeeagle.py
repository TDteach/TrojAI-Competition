import os
import logging

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

import torchvision
from torchvision.models.detection.ssd import SSDFeatureExtractorVGG, SSD

import skimage
import numpy as np
from tqdm import tqdm


class SSDBackboneHijacker(nn.Module):
    def __init__(self, backbone: SSDFeatureExtractorVGG):
        super().__init__()
        self.backbone = backbone
        self.record_features = False
        self.last_features = None

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone.features(x)
        return self.forward_features(x)

    def forward_features(self, features: Tensor) -> Dict[str, Tensor]:
        x = features

        if self.record_features:
            self.last_features = torch.clone(x)

        rescaled = self.backbone.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        # Calculating Feature maps for the rest blocks
        for block in self.backbone.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


class SSDHijacker(nn.Module):
    def __init__(self, ssd: SSD):
        super().__init__()
        self.backbone = SSDBackboneHijacker(ssd.backbone)
        ssd.backbone = self.backbone

        self.ssd = ssd

    def set_record_features(self, record_features: bool = None):
        if record_features is None:
            self.backbone.record_features = True
        else:
            self.backbone.record_features = record_features

    def forward(
            self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None,
            record_features: bool = False
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        return self.ssd(images, targets)

    def get_last_features(self) -> Tensor:
        assert self.backbone.last_features is not None, "No features recorded"
        return self.backbone.last_features

    def get_head_outputs(
            self, images: Optional[List[Tensor]] = None,
            features_replace: Optional[Union[Tensor, Variable]] = None,
    ) -> List[Tensor]:
        if images is not None:
            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))

            # transform the input
            images, targets = self.ssd.transform(images, targets=None)

            features = self.ssd.backbone(images.tensors)
        elif features_replace is not None:
            features = self.backbone.forward_features(features_replace)
        else:
            raise ValueError("Either images or features_replace must be provided")

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        head_outputs = self.ssd.head(features)

        return head_outputs


def load_examples(examples_dirpath):
    # Augmentation transformations
    augmentation_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ConvertImageDtype(torch.float)])

    image_list = list()
    logging.info("Loading clean example images.")
    # Inference on models
    for examples_dir_entry in os.scandir(examples_dirpath):
        if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
            # load the example image
            img = skimage.io.imread(examples_dir_entry.path)

            # convert the image to a tensor
            # should be uint8 type, the conversion to float is handled later
            image = torch.as_tensor(img)

            # move channels first
            image = image.permute((2, 0, 1))

            # convert to float (which normalizes the values)
            image = augmentation_transforms(image)
            image_list.append(image)
            # image = image.to(device)
    image_list = torch.stack(image_list, dim=0)

    return image_list


def detect(model_filepath: str, examples_dirpath: str, per_class: int = 2, batch_size: int = 400, max_iters=1000):
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info("Using compute device: {}".format(device))

    example_images = load_examples(examples_dirpath)

    model = torch.load(model_filepath)
    model_class = model.__class__.__name__
    num_classes = model.head.classification_head.num_columns
    model.eval()
    model.to(device)

    intl_c = batch_size // per_class
    if intl_c * per_class < batch_size:
        intl_c += 1
        batch_size = intl_c * per_class

    new_model = SSDHijacker(model)
    new_model.set_record_features()
    outputs = new_model(example_images.to(device))
    last_features = new_model.backbone.last_features
    new_model.set_record_features(False)

    max_v = torch.max(last_features).item()
    std_v = torch.std(last_features.flatten()).item()
    mean_v = torch.mean(last_features.flatten()).item()
    print(std_v, mean_v)

    # common_features = last_features[:per_class, :, :, :]
    idx = np.random.choice(len(last_features), per_class)
    common_features = last_features[idx, :, :, :]

    rst_list = []

    # while num_classes is None or target_class < num_classes:
    for st_c in range(0, num_classes, intl_c):
        new_model.eval()

        intl_r = min(intl_c, num_classes - st_c)
        batch_size = intl_r * per_class
        print('deal classes [', st_c, ',', st_c + intl_r, ')', '-' * 20)

        # idx = np.random.choice(len(last_features), batch_size)
        # init_features = last_features[idx, :]
        init_features = common_features.repeat([intl_r, 1, 1, 1])

        init_values = torch.randn_like(init_features) * std_v + mean_v
        # init_values = torch.rand_like(init_features) + init_features
        # init_values = torch.rand_like(init_features)
        var = Variable(init_values, requires_grad=True)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([var], lr=1e-2, weight_decay=0.005)

        label = torch.ones(batch_size, dtype=torch.long, device=device)
        for st_i in range(0, batch_size, per_class):
            tgt_class = st_i // per_class + st_c
            label[st_i:st_i + per_class] *= tgt_class

        best_loss = None
        best_var = None
        for iters in tqdm(range(max_iters)):
            optimizer.zero_grad()
            head_outputs = new_model.get_head_outputs(features_replace=var)
            cls_logits = head_outputs['cls_logits']
            # if num_classes is None:
            #     num_classes = cls_logits.shape[-1]
            cls_logits = torch.reshape(cls_logits, (-1, num_classes))
            if label.shape[0] < cls_logits.shape[0]:
                t = cls_logits.shape[0] // label.shape[0]
                label = label.unsqueeze(1)
                label = label.repeat((1, t))
                label = label.flatten()
                assert label.shape[0] == cls_logits.shape[0], "Dimensions mismatched"
            loss_all = criterion(cls_logits, label)
            loss_all = torch.reshape(loss_all, (batch_size, -1))
            loss_min = torch.min(loss_all, dim=1)

            loss = torch.sum(loss_min.values)

            # early stop
            z = loss.item()
            if best_loss is None or z < best_loss:
                best_loss = z
                best_var = var.data.clone()
                # print('update best_loss to', z)
            if z < 1e-6:
                break

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                var = var.clamp_(0, 999.)
            # print(iters, loss.item())

        with torch.no_grad():
            head_outputs = new_model.get_head_outputs(features_replace=best_var)
            cls_logits = head_outputs['cls_logits']
            cls_probs = torch.softmax(cls_logits, dim=-1)
            a = []
            for st_i in range(0, batch_size, per_class):
                tgt_class = st_i // per_class + st_c
                _probs = cls_probs[st_i:st_i + per_class, :, :]
                rst = torch.max(_probs[:, :, tgt_class], axis=-1)
                for k, i in enumerate(rst.indices):
                    a.append(_probs[k, i, :].detach().cpu().numpy())
            a = np.reshape(a, (intl_r, per_class, -1))
            mean_probs = np.mean(a, axis=1)
        rst_list.append(mean_probs)

    rst_mat = np.concatenate(rst_list, axis=0)

    return rst_mat

    print(rst_mat.shape)
    with open('rst_mat.npy', 'wb') as f:
        np.save(f, rst_mat)

    for i in range(len(rst_mat)):
        rst_mat[i, i] = 0
    mean_rst = np.mean(rst_mat, axis=1)
    ord = np.argsort(mean_rst)
    for k, o in enumerate(ord):
        print(k, o, mean_rst[o])

    exit(0)
    return rst_mat


if __name__ == '__main__':
    pass
