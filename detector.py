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
from utils.flatten import flatten_model, flatten_models, regularize_model_parameters
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
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GroupKFold

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

import lightgbm as lgb

from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

from scipy.special import softmax, log_softmax
from scipy import stats
from tqdm import tqdm
import time

from typing import Optional
from pprint import pprint

import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
import sklearn.metrics

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def ext_quantiles(a, bins=100):
    qs = [i / bins for i in range(bins)]
    return np.quantile(a, qs)


def get_forward_hook_fn(k, in_list, out_list):
    def hook_fn(module, input, output):
        in_list[k] = input[0].detach().cpu().numpy()
        # out_list[k] = output.detach().cpu().numpy()
        out_list[k] = output

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
    preds = torch.argmax(logits, axis=-1)
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
    assert len(X) == len(Y)

    n = len(X)
    d = (n - 1) ** 2

    mX = X - np.mean(X, axis=0)
    mY = Y - np.mean(Y, axis=0)

    XX = np.matmul(mX, mX.transpose())
    YY = np.matmul(mY, mY.transpose())

    u = np.trace(np.matmul(XX, YY)) / d
    d1 = np.trace(np.matmul(XX, XX)) / d
    d2 = np.trace(np.matmul(YY, YY)) / d

    return u / np.sqrt(d1 * d2)


def extract_aligned_features(model_feats, ref_feats):
    src_repr, tgt_repr = model_feats['model_repr'], ref_feats['model_repr']
    src_repr_names, tgt_repr_names = list(src_repr.keys()), list(tgt_repr.keys())
    src_ins, tgt_ins = model_feats['runtime_feats']['mid_ins'], ref_feats['runtime_feats']['mid_ins']
    src_outs, tgt_outs = model_feats['runtime_feats']['mid_outs'], ref_feats['runtime_feats']['mid_outs']
    src_grads, tgt_grads = model_feats['runtime_feats']['mid_grads'], ref_feats['runtime_feats']['mid_grads']

    g0, g1 = np.mean(src_grads[-2][:10, :], axis=0), np.mean(src_grads[-1][:10, :],
                                                             axis=0)  # 0 w.r.t good,  1 w.r.t good
    g2, g3 = np.mean(src_grads[-2][10:, :], axis=0), np.mean(src_grads[-1][10:, :],
                                                             axis=0)  # 0 w.r.t bad,   1 w.r.t bad

    gg = np.asarray([g0, g1, g2, g3])
    Z = np.matmul(gg, gg.transpose())

    g0 /= np.linalg.norm(g0)
    g1 /= np.linalg.norm(g1)
    g2 /= np.linalg.norm(g2)
    g3 /= np.linalg.norm(g3)
    gg = np.stack([g0, g1, g2, g3]).reshape(1, -1)

    a = None
    for i in range(0, len(src_repr_names), 2):
        w = src_repr[src_repr_names[i]]
        if a is None:
            a = w.transpose()
        else:
            a = np.matmul(a, w.transpose())
    z = a[:, 0]
    z /= np.linalg.norm(z)
    feats = z.reshape(1, -1)

    # return feats

    # ggg = np.asarray([g2[8], g3[8], feats[0,8], Z[3,3], Z[3,2], Z[2,2]])
    # return np.expand_dims(ggg, axis=0)

    ggg = np.concatenate([gg, Z.reshape(1, -1), feats], axis=1)
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


def select_reference_models(num, md_id, nd):
    output = []
    all_n = nd.keys()
    for i in range(num):
        b = []
        for n in all_n:
            rid = md_id
            while rid == md_id:
                rid = np.random.choice(nd[n], 1)[0]
            b.append(rid)
        output.append(b)
    return output


def extract_ks_features(model, ref_models, feat_inds=None):
    ind = -1
    feat = []
    cnt = 0
    if feat_inds is not None: n_feat = len(feat_inds)
    for ref_model in ref_models:
        for layer in model.keys():
            ow, rw = model[layer], ref_model[layer]
            if feat_inds is not None and len(ow) + ind < feat_inds[cnt]:
                ind += len(ow)
                continue

            for o, r in zip(ow, rw):
                ind += 1
                if feat_inds is not None and ind < feat_inds[cnt]:
                    continue
                cnt += 1
                rst = stats.kstest(o, r)
                feat.append(rst.statistic)
                if feat_inds is not None and cnt >= n_feat:
                    break
            if feat_inds is not None and cnt >= n_feat:
                break
        if feat_inds is not None and cnt >= n_feat:
            break

    feat = np.reshape(feat, (1, cnt))
    return feat


def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


class MutualInfoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, input_features, random_state=None):
        # self.input_features = 100
        # self.f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
        self.input_features = input_features
        self.f_selector = None
        self.random_state = random_state

    def fit(self, X, Y=None):
        self.f_selector = SelectKBest(mutual_info_classif, k=100)
        self.f_selector.fit(X, y)
        return self

    def transform(self, X):
        XX = self.f_selector.transform(X)
        return XX

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MutualInfoPreprocessing",
            "name": "MutualInfoPreprocessing",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cf = ConfigurationSpace()  # Return an empty configuration as there is None
        input_features = UniformIntegerHyperparameter(
            name="input_features", lower=32, upper=128, default_value=100,
        )
        cf.add_hyperparameters([input_features])
        return cf


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
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.ref_model_dict_filepath = os.path.join(self.learned_parameters_dirpath, "ref_model_dict.pkl")

        self.train_seed = metaparameters["train_seed"]
        self.train_data_augment_factor = metaparameters["train_data_augment_factor"]
        self.input_features = metaparameters["train_input_features"]
        self.automl_num_folds = metaparameters["train_automl_num_folds"]
        self.automl_kwargs = {
            'time_left_for_this_task': metaparameters["train_automl_time_left_for_this_task"],
            'n_jobs': metaparameters["train_automl_n_jobs"],
            'memory_limit': metaparameters["train_automl_memory_limit"] * 1024,
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_seed": self.train_seed,
            "train_automl_num_folds": self.automl_num_folds,
            "train_automl_time_left_for_this_task": self.automl_kwargs['time_left_for_this_task'],
            "train_automl_n_jobs": self.automl_kwargs['n_jobs'],
            "train_automl_memory_limit": self.automl_kwargs['memory_limit'],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)),
                  "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """

        autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MutualInfoPreprocessing)

        archs = ['FasterRCNN', 'DetrForObjectDetection', 'SSD']
        # archs = ['DetrForObjectDetection', 'SSD']
        # archs = ['FasterRCNN']
        for model_arch in archs:
            with open(f'train_dataset_{model_arch}.pkl', 'rb') as fp:
                dataset = pickle.load(fp)

            X = [data[0] for data in dataset]
            y = [data[1] for data in dataset]
            X = np.concatenate(X, axis=0)
            y = np.asarray(y)
            print(X.shape)
            print(y.shape)

            # '''
            f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
            f_selector.fit(X, y)
            XX = f_selector.transform(X)
            print(XX.shape)
            X = XX
            # '''

            n = len(X) // self.train_data_augment_factor
            a = np.arange(n)
            a = np.tile(a, (self.train_data_augment_factor, 1))
            a = a.T.flatten()
            resampling_strategy = GroupKFold(n_splits=self.automl_num_folds)
            automl = autosklearn.classification.AutoSklearnClassifier(
                # include={"feature_preprocessor":["MutualInfoPreprocessing"]},
                metric=autosklearn.metrics.roc_auc,
                resampling_strategy=resampling_strategy,
                resampling_strategy_arguments={'folds': self.automl_num_folds, 'groups': a},
                **self.automl_kwargs,
            )
            print('automl has been set up')
            automl.fit(X, y)

            print(automl.leaderboard(ensemble_only=False))
            # pprint(automl.show_models(), indent=4)
            print(automl.sprint_statistics())
            automl.refit(X, y)

            model = {
                'f_selector': f_selector,
                'automl': automl,
            }

            model_filepath = os.path.join(self.learned_parameters_dirpath, f'automl_model_{model_arch}.pkl')
            with open(model_filepath, 'wb') as fh:
                pickle.dump(model, fh)

            with open(model_filepath, 'rb') as fh:
                model = pickle.load(fh)
            automl = model['automl']

            y_pred = automl.predict(X)
            print(f'Training for {model_arch} AUC:', accuracy_score(y, y_pred))

        self.write_metaparameters()
        logging.info("Configuration done!")


    def get_data_from_FE_results(self, rst_list):
        X, y = [], []
        for rst in rst_list:
            model_info = rst['model_info']
            rst_mat = rst['rst_mat']

            lb = model_info['ground_truth']
            y.append(lb)

            fet = []
            flatten_mat = rst_mat.flatten()
            fet.append(np.mean(flatten_mat))
            fet.append(np.std(flatten_mat))

            no = np.linalg.norm(flatten_mat)
            fet.append(no)

            # normed_mat = flatten_mat / no
            normed_mat = flatten_mat
            a = ext_quantiles(normed_mat, bins=200)
            fet.extend(a)
            fet.append(np.max(normed_mat))
            X.append(fet)

        return np.asarray(X), np.asarray(y)


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

        # '''
        model_repr_dict, model_ground_truth_dict, model_info_dict = load_models_dirpath(model_path_list, return_info=True)

        # print(list(model_repr_dict.keys()))
        model_name = 'FasterRCNN'
        # model_name = 'DetrForObjectDetection'
        model_repr_list = model_repr_dict[model_name]
        model_ground_truth = model_ground_truth_dict[model_name]

        model_info_list = model_info_dict[model_name]
        for fo in model_info_list:
            if fo['trigger_type'] is not None:
                print(fo)
        exit(0)

        y_true, y_score = list(), list()
        for model_repr, lb in zip(model_repr_list, model_ground_truth):
            if model_name == 'FasterRCNN':
                w = model_repr['roi_heads.box_predictor.cls_score.weight']
                b = model_repr['roi_heads.box_predictor.cls_score.bias']
            elif model_name == 'DetrForObjectDetection':
                w = model_repr['class_labels_classifier.weight']
                b = model_repr['class_labels_classifier.bias']
            num_classes = len(b)
            nw = np.sum(np.square(w), axis=1)
            z = b + 0.5*nw
            pi_z = softmax(z)
            log_pi_z = log_softmax(z)
            sorted_pi_z = np.sort(pi_z)
            sorted_log_pi_z = np.sort(log_pi_z)
            log_pi_z += np.log(num_classes)

            y_true.append(lb)
            y_score.append(np.std(pi_z))

            print(lb, np.mean(pi_z), np.std(pi_z), np.max(pi_z)-np.min(pi_z), (sorted_pi_z[-1]-sorted_pi_z[-2])*num_classes)
            # if lb == 0 and np.min(pi_z) < 1e-4:
            #     print(num_classes)
            #     print(log_pi_z)

        auc = roc_auc_score(y_true, y_score)
        print(auc)

        exit(0)

        from freeeagle import detect as freeeagle_detect

        for model_arch, models_info in model_info_dict.items():
            print(model_arch)
            if model_arch != 'SSD': continue

            rst_list = []
            for model_info in tqdm(models_info):
                model_path, lb = model_info['model_path'], model_info['ground_truth']
                print(model_path, lb)

                head, tail = os.path.split(model_path)
                examples_dirpath = os.path.join(head, 'clean-example-data')

                rst_mat = freeeagle_detect(model_path, examples_dirpath)

                rst_list.append({
                    'model_info' : model_info,
                    'rst_mat' : rst_mat,
                })

            with open('freeeagle_mat.pkl', 'wb') as f:
                pickle.dump(rst_list, f)
        # '''

        with open('freeeagle_mat.pkl', 'rb') as f:
            rst_list = pickle.load(f)

        X, y = self.get_data_from_FE_results(rst_list)
        print(X.shape, y.shape)

        self.automl_kwargs['time_left_for_this_task'] = 60
        print(self.automl_kwargs)

        a = np.arange(len(X))
        num_folds = 4
        resampling_strategy = GroupKFold(n_splits=num_folds)
        automl = autosklearn.classification.AutoSklearnClassifier(
                # include={"feature_preprocessor":["MutualInfoPreprocessing"]},
                metric=autosklearn.metrics.roc_auc,
                resampling_strategy=resampling_strategy,
                resampling_strategy_arguments={'folds': num_folds, 'groups': a},
                **self.automl_kwargs,
        )
        print('automl has been set up')
        automl.fit(X, y)

        print(automl.leaderboard(ensemble_only=False))
        # pprint(automl.show_models(), indent=4)
        print(automl.sprint_statistics())



        exit(0)

        ncls_dict = dict()
        for model_arch, models_info in model_info_dict.items():
            a = dict()
            for k, model_info in enumerate(models_info):
                n_classes = model_info['n_classes']
                if n_classes not in a:
                    a[n_classes] = []
                a[n_classes].append(k)
            ncls_dict[model_arch] = a
        print(ncls_dict)
        # delete those n-classes classifiers with only one instance
        for model_arch, a in ncls_dict.items():
            b = []
            for n in a.keys():
                if len(a[n]) < 2:
                    b.append(n)
            for nn in b:
                del a[nn]
        print(ncls_dict)

        '''
        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)
        print(models_padding_dict)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        check_models_consistency(model_repr_dict)
        '''

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

        # with open("layer_transform.pkl", "rb") as fh:
        #    layer_transform = pickle.load(fh)
        # logging.info("Feature reduction applied. Creating feature file...")

        ref_model_dict = {}

        for model_arch, models in flat_models.items():
            X = None
            y = []

            logging.info("Parsing %s models...", model_arch)
            for md_id in tqdm(range(len(models))):
                ref_model_ids = select_reference_models(self.train_data_augment_factor, md_id, ncls_dict[model_arch])

                if model_arch == 'FasterRCNN':
                    break

                for model_ids in ref_model_ids:
                    model_feats = extract_ks_features(models[md_id], [models[i] for i in model_ids])
                    if X is None:
                        X = model_feats
                    else:
                        X = np.vstack((X, model_feats))
                    y.append(model_ground_truth_dict[model_arch][md_id])

            ref_model_dict[model_arch] = [models[i] for i in ref_model_ids[0]]
            if model_arch == 'FasterRCNN':
                continue

            print(X.shape)

            dataset = []
            for _x, _y in zip(X, y):
                _x = np.expand_dims(_x, axis=0)
                dataset.append((_x, _y))

            with open(f'train_dataset_{model_arch}.pkl', 'wb') as fp:
                pickle.dump(dataset, fp)

        with open(self.ref_model_dict_filepath, "wb") as fh:
            pickle.dump(ref_model_dict, fh)

        return

        '''
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
        # '''

        logging.info("Building RandomForest based on random features, with the provided mean and std.")
        rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        X = []
        y = []
        for model_arch in model_repr_dict.keys():
            for model_index in range(len(model_repr_dict[model_arch])):
                y.append(model_ground_truth_dict[model_arch][model_index])

                model_feats = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'],
                                         size=(1, self.input_features))
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

        from utils.show import display_objdetect_image

        print('inference on ', examples_dirpath)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        # move the model to the GPU in eval mode
        model.to(device)
        model.eval()

        # Augmentation transformations
        augmentation_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ConvertImageDtype(torch.float)])

        # '''
        cls_head = model.roi_heads.box_predictor.cls_score
        in_list = [None]*1
        out_list = [None]*1
        cls_head.register_forward_hook(get_forward_hook_fn(0, in_list, out_list))
        # '''


        logging.info("Evaluating the model on the clean example images.")
        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                # load the example image
                path = examples_dir_entry.path

                if not path.endswith('13.png'):
                    continue


                img = skimage.io.imread(path)
                img_name = os.path.split(path)[-1]
                print(path)

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


                # show hook
                print(in_list[0][235])
                print(in_list[0].shape)

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

                ground_truth_filepath = examples_dir_entry.path.replace('.png', '.json')

                with open(ground_truth_filepath, mode='r', encoding='utf-8') as f:
                    ground_truth = jsonpickle.decode(f.read())

                logging.info(
                    "Model predicted {} boxes, Ground Truth has {} boxes.".format(len(pred), len(ground_truth)))
                # logging.info("Model: {}, Ground Truth: {}".format(examples_dir_entry.name, ground_truth))

                image = examples_dir_entry.path
                display_objdetect_image(image, boxes, labels, scores, out_name=f'out_{img_name}', score_top=len(ground_truth))
                print(labels)
                print(scores)

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
        os.environ['MPLCONFIGDIR'] = os.path.abspath(scratch_dirpath)
        print(os.getenv('MPLCONFIGDIR'))

        st_time = time.time()

        model, model_repr, model_class = load_model(model_filepath)

        self.inference_on_example_data(model, examples_dirpath)
        # from freeeagle import detect as freeeagle_detect
        # freeeagle_detect(model_filepath, examples_dirpath)

        exit(0)

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        # with open(self.models_padding_dict_filepath, "rb") as fp:
        #    model_padding_dict = pickle.load(fp)

        # load the model
        # flat_model = flatten_model(model_repr, model_layer_map[model_class])
        flat_model = regularize_model_parameters(model_repr, model_layer_map[model_class])

        ed_time = time.time()
        print('flat time:', ed_time - st_time)

        model_filepath = os.path.join(self.learned_parameters_dirpath, f'automl_model_{model_class}.pkl')
        with open(model_filepath, 'rb') as fh:
            model = pickle.load(fh)

        f_selector = model['f_selector']
        automl = model['automl']

        mask = f_selector.get_support()
        feat_inds = np.arange(mask.shape[0])[mask]

        with open(self.ref_model_dict_filepath, "rb") as fh:
            ref_model_dict = pickle.load(fh)

        X = extract_ks_features(flat_model, ref_model_dict[model_class], feat_inds=feat_inds)
        # X = extract_ks_features(flat_model, ref_model_dict[model_class], feat_inds=None)
        # X = f_selector.transform(X)
        print(X.shape)

        ed_time = time.time()
        print('feature extraction time:', ed_time - st_time)

        probability = automl.predict_proba(X)[0][1]
        # clip the probability to reasonable values
        # probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
        ed_time = time.time()
        print(ed_time - st_time)
