# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath, create_layer_map
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)

import torch
import torchvision
import skimage.io


import copy
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, log_loss

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

import lightgbm as lgb

from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

from scipy.special import softmax
from tqdm import tqdm


def ext_quantiles(a, bins=100):
    qs = [i/bins for i in range(bins)]
    return np.quantile(a, qs)


def get_forward_hook_fn(k, in_list, out_list):
    def hook_fn(module, input, output):
        in_list[k] = input[0].detach().cpu().numpy()
        out_list[k] = output.detach().cpu().numpy()

    return hook_fn

def get_backward_hook_fn(k, grad_list):
    def hook_fn(module, grad_input, grad_output):
        grad_list[k] = grad_output[0].detach().cpu().numpy()

    return hook_fn

def extract_runtime_features(model, inputs):
    child_list = list(model.children())
    n_child = len(child_list)
    mid_in_list = [None] * n_child
    mid_out_list = [None] * n_child
    grad_list = [None] * n_child
    hook_handler_list = list()
    for k, m in enumerate(model.children()):
        handler = m.register_forward_hook(get_forward_hook_fn(k, mid_in_list, mid_out_list))
        hook_handler_list.append(handler)
        handler = m.register_full_backward_hook(get_backward_hook_fn(k, grad_list))
        hook_handler_list.append(handler)

    label_tensor = torch.ones(len(inputs), dtype=torch.int64) * 0
    input_tensor = torch.from_numpy(inputs)
    input_variable = Variable(input_tensor, requires_grad=True)
    logits = model(input_variable)
    preds = torch.argmax(logits,axis=-1)
    loss = F.cross_entropy(logits, label_tensor)
    loss.backward()

    for handler in hook_handler_list:
        handler.remove()

    grad_list.append(input_variable.grad.detach().cpu().numpy())

    label_tensor = torch.ones(len(inputs), dtype=torch.int64) * 1
    input_tensor = torch.from_numpy(inputs)
    input_variable = Variable(input_tensor, requires_grad=True)
    logits = model(input_variable)
    loss = F.cross_entropy(logits, label_tensor)
    loss.backward()
    grad_list.append(input_variable.grad.detach().cpu().numpy())

    para_grads = OrderedDict(
        {na: w.grad.detach().cpu().numpy() for (na, w) in model.named_parameters()}
    )

    return {
        'mid_ins': mid_in_list,
        'mid_outs': mid_out_list,
        'mid_grads': grad_list,
        'para_grads': para_grads,
    }



def CKA(X, Y):
    assert len(X)==len(Y)

    n = len(X)
    d = (n-1)**2

    mX = X - np.mean(X, axis=0)
    mY = Y - np.mean(Y, axis=0)

    XX = np.matmul(mX, mX.transpose())
    YY = np.matmul(mY, mY.transpose())

    u = np.trace(np.matmul(XX,YY))/d
    d1 = np.trace(np.matmul(XX,XX))/d
    d2 = np.trace(np.matmul(YY,YY))/d

    return u/np.sqrt(d1*d2)


def extract_aligned_features(model_feats, ref_feats):
    src_repr, tgt_repr = model_feats['model_repr'], ref_feats['model_repr']
    src_repr_names, tgt_repr_names = list(src_repr.keys()), list(tgt_repr.keys())
    src_ins, tgt_ins = model_feats['runtime_feats']['mid_ins'], ref_feats['runtime_feats']['mid_ins']
    src_outs, tgt_outs = model_feats['runtime_feats']['mid_outs'], ref_feats['runtime_feats']['mid_outs']
    src_grads, tgt_grads = model_feats['runtime_feats']['mid_grads'], ref_feats['runtime_feats']['mid_grads']


    g0, g1 = np.mean(src_grads[-2][:10, :],axis=0), np.mean(src_grads[-1][:10, :],axis=0) # 0 w.r.t good,  1 w.r.t good
    g2, g3 = np.mean(src_grads[-2][10:, :],axis=0), np.mean(src_grads[-1][10:, :],axis=0) # 0 w.r.t bad,   1 w.r.t bad

    gg = np.asarray([g0,g1,g2,g3])
    Z = np.matmul(gg, gg.transpose())

    g0 /= np.linalg.norm(g0)
    g1 /= np.linalg.norm(g1)
    g2 /= np.linalg.norm(g2)
    g3 /= np.linalg.norm(g3)
    gg = np.stack([g0, g1, g2, g3]).reshape(1,-1)


    a = None
    for i in range(0, len(src_repr_names), 2):
        w = src_repr[src_repr_names[i]]
        if a is None:
            a = w.transpose()
        else:
            a = np.matmul(a, w.transpose())
    z = a[:, 0]
    z /= np.linalg.norm(z)
    feats = z.reshape(1,-1)

    #return feats


    #ggg = np.asarray([g2[8], g3[8], feats[0,8], Z[3,3], Z[3,2], Z[2,2]])
    #return np.expand_dims(ggg, axis=0)


    ggg = np.concatenate([gg, Z.reshape(1,-1), feats], axis=1)
    return ggg


def feature_extraction(model, inputs, model_repr=None, model_class=None, ref_feats=None):
    if isinstance(model, str):
        model = torch.load(model)
        model.eval()

    if model_repr is None:
        model_repr = OrderedDict(
            {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
        )
    if model_class is None:
        model_class = model.__class__.__name__

    runtime_feats = extract_runtime_features(model, inputs)
    model_feats = {
        'model_repr': model_repr,
        'runtime_feats': runtime_feats,
        'model_class': model_class,
    }

    if ref_feats is None:
        return model_feats

    aligned_feats = extract_aligned_features(model_feats, ref_feats)
    return aligned_feats








def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = os.path.join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = os.path.join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.input_features = metaparameters["train_input_features"]
        self.weight_params = {
            "rso_seed": metaparameters["train_weight_rso_seed"],
            "mean": metaparameters["train_weight_params_mean"],
            "std": metaparameters["train_weight_params_std"],
        }
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_rso_seed"],
            "mean": metaparameters["train_weight_params_mean"],
            "std": metaparameters["train_weight_params_std"],
            "scaler": 1.0,
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_rso_seed": self.weight_params["rso_seed"],
            "train_weight_params_mean": self.weight_params["mean"],
            "train_weight_params_std": self.weight_params["std"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))


    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """

        with open('train_dataset.pkl','rb') as fp:
            dataset = pickle.load(fp)

        X = [data[0] for data in dataset]
        y = [data[1] for data in dataset]
        X = np.concatenate(X, axis=0)
        y = np.asarray(y)
        print(X.shape)
        print(y.shape)

        import autosklearn.classification
        import autosklearn.metrics
        from sklearn.metrics import accuracy_score

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=10800,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 4},
            n_jobs=32,
            memory_limit=1024*32,
            metric=autosklearn.metrics.roc_auc,
        )
        automl.fit(X,y)

        print(automl.leaderboard(ensemble_only=False))
        print(automl.sprint_statistics())

        model_filepath = os.path.join(self.learned_parameters_dirpath, 'automl_model.pkl')
        with open(model_filepath, 'wb') as fh:
            pickle.dump(automl, fh)

        with open(model_filepath, 'rb') as fh:
            automl = pickle.load(fh)

        y_pred = automl.predict(X)
        print(accuracy_score(y, y_pred))

        self.write_metaparameters()
        logging.info("Configuration done!")



    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)



        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        check_models_consistency(model_repr_dict)

        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        '''
        # Flatten models
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        with open("flat_models.pkl", "wb") as fh:
            pickle.dump(flat_models, fh)
        # '''

        with open("flat_models.pkl", "rb") as fh:
            flat_models = pickle.load(fh)

        logging.info("Models flattened. Fitting feature reduction...")

        '''
        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)
        with open("layer_transform.pkl", "wb") as fh:
            pickle.dump(layer_transform, fh)
        # '''

        with open("layer_transform.pkl", "rb") as fh:
            layer_transform = pickle.load(fh)


        logging.info("Feature reduction applied. Creating feature file...")
        X = None
        y = []

        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            logging.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1

                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_arch], model
                )
                print(model_feats.shape)
                if X is None:
                    X = model_feats
                else:
                    X = np.vstack((X, model_feats))

        print(X.shape)

        dataset = []
        for _x, _y in zip(X, y):
            _x = np.expand_dims(_x, axis=0)
            dataset.append((_x, _y))

        with open('train_dataset.pkl','wb') as fp:
            pickle.dump(dataset, fp)

        exit(0)














        logging.info("Building RandomForest based on random features, with the provided mean and std.")
        rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        X = []
        y = []
        for model_arch in model_repr_dict.keys():
            for model_index in range(len(model_repr_dict[model_arch])):
                y.append(model_ground_truth_dict[model_arch][model_index])

                model_feats = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1,self.input_features))
                X.append(model_feats)
        X = np.vstack(X)

        logging.info("Training RandomForestRegressor model.")
        model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")


    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        # move the model to the GPU in eval mode
        model.to(device)
        model.eval()

        # Augmentation transformations
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        logging.info("Evaluating the model on the clean example images.")
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
                image = image.to(device)

                # Convert to NCHW
                image = image.unsqueeze(0)

                # inference
                outputs = model(image)
                # handle multiple output formats for different model types
                if 'DetrObjectDetectionOutput' in outputs.__class__.__name__:
                    # DETR doesn't need to unpack the batch dimension
                    boxes = outputs.pred_boxes.cpu().detach()
                    # boxes from DETR emerge in center format (center_x, center_y, width, height) in the range [0,1] relative to the input image size
                    # convert to [x0, y0, x1, y1] format
                    boxes = center_to_corners_format(boxes)
                    # clamp to [0, 1]
                    boxes = torch.clamp(boxes, min=0, max=1)
                    # and from relative [0, 1] to absolute [0, height] coordinates
                    img_h = img.shape[0] * torch.ones(1)  # 1 because we only have 1 image in the batch
                    img_w = img.shape[1] * torch.ones(1)
                    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                    boxes = boxes * scale_fct[:, None, :]

                    # unpack the logits to get scores and labels
                    logits = outputs.logits.cpu().detach()
                    prob = torch.nn.functional.softmax(logits, -1)
                    scores, labels = prob[..., :-1].max(-1)

                    boxes = boxes.numpy()
                    scores = scores.numpy()
                    labels = labels.numpy()

                    # all 3 items have a batch size of 1 in the front, so unpack it
                    boxes = boxes[0,]
                    scores = scores[0,]
                    labels = labels[0,]
                else:
                    # unpack the batch dimension
                    outputs = outputs[0]  # unpack the batch size of 1
                    # for SSD and FasterRCNN outputs are a list of dict.
                    # each boxes is in corners format (x_0, y_0, x_1, y_1) with coordinates sized according to the input image

                    boxes = outputs['boxes'].cpu().detach().numpy()
                    scores = outputs['scores'].cpu().detach().numpy()
                    labels = outputs['labels'].cpu().detach().numpy()

                # wrap the network outputs into a list of annotations
                pred = utils.models.wrap_network_prediction(boxes, labels)

                # logging.info('example img filepath = {}, Pred: {}'.format(examples_dir_entry.name, pred))

                ground_truth_filepath = examples_dir_entry.path.replace('.png','.json')

                with open(ground_truth_filepath, mode='r', encoding='utf-8') as f:
                    ground_truth = jsonpickle.decode(f.read())

                logging.info("Model predicted {} boxes, Ground Truth has {} boxes.".format(len(pred), len(ground_truth)))
                # logging.info("Model: {}, Ground Truth: {}".format(examples_dir_entry.name, ground_truth))


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        with open(self.models_padding_dict_filepath, "rb") as fp:
            model_padding_dict = pickle.load(fp)

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        with open("layer_transform.pkl", "rb") as fp:
            layer_transform = pickle.load(fp)

        # load the model
        model, model_repr, model_class = load_model(model_filepath)
        model_repr = pad_model(model_repr, model_class, model_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])

        X = use_feature_reduction_algorithm(layer_transform[model_class], flat_model)
        print(X.shape)

        model_filepath = os.path.join(self.learned_parameters_dirpath, 'automl_model.pkl')
        with open(model_filepath, 'rb') as fh:
            automl = pickle.load(fh)

        probability = automl.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
