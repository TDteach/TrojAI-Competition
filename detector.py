import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)

from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch

from gather_data import read_all_examples

import copy
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

import lightgbm as lgb

from collections import OrderedDict


def ext_quantiles(a, bins=100):
    qs = [i/bins for i in range(bins)]
    return np.quantile(a, qs)


def align_weight(weight, target):
    flat_weight = weight.flatten()
    flat_target = target.flatten()
    mean_weight = np.mean(flat_weight)
    std_weight = np.std(flat_weight)
    mean_target = np.mean(flat_target)
    std_target = np.std(flat_target)

    weight = (weight-mean_weight)/std_weight
    target = (target-mean_target)/std_target


    w_T = np.transpose(weight)
    Z = np.matmul(w_T, target)
    u, s, vh = np.linalg.svd(Z)
    H = np.matmul(u[:, :vh.shape[0]], vh)

    trans_w = np.matmul(weight, H)

    # trans_w = trans_w*std_weight + mean_weight

    '''
    w_inv = np.linalg.pinv(weight)
    M = np.matmul(w_inv, target)
    trans_w = np.matmul(weight, M)
    '''
    return trans_w



def get_forward_hook_fn(k, embed_list):
    def hook_fn(module, input, output):
        embed_list[k] = output.detach().cpu().numpy()

    return hook_fn

def get_backward_hook_fn(k, grad_list):
    def hook_fn(module, grad_input, grad_output):
        grad_list[k] = grad_output[0].detach().cpu().numpy()

    return hook_fn

def extract_runtime_features(model, inputs):
    child_list = list(model.children())
    n_child = len(child_list)
    embed_list = [None] * n_child
    grad_list = [None] * n_child
    hook_handler_list = list()
    for k, m in enumerate(model.children()):
        handler = m.register_forward_hook(get_forward_hook_fn(k, embed_list))
        hook_handler_list.append(handler)
        handler = m.register_backward_hook(get_backward_hook_fn(k, grad_list))
        hook_handler_list.append(handler)

    label_tensor = torch.ones(len(inputs), dtype=torch.int64)
    input_tensor = torch.from_numpy(inputs)
    input_variable = Variable(input_tensor, requires_grad=True)
    logits = model(input_variable)
    preds = torch.argmax(logits,axis=-1)
    loss = F.cross_entropy(logits, label_tensor)
    loss.backward()

    for handler in hook_handler_list:
        handler.remove()

    embed_list.append(inputs)
    grad_list.append(input_variable.grad.detach().cpu().numpy())

    para_grads = OrderedDict(
        {na: w.grad.detach().cpu().numpy() for (na, w) in model.named_parameters()}
    )

    return {
        'mid_outs': embed_list,
        'mid_grads': grad_list,
        'para_grads': para_grads,
    }


def extract_aligned_features(model_feats, ref_feats):
    src_repr, tgt_repr = model_feats['model_repr'], ref_feats['model_repr']
    src_outs, tgt_outs = model_feats['runtime_feats']['mid_outs'], ref_feats['runtime_feats']['mid_outs']

    src_o, tgt_o = src_outs[0], tgt_outs[0]

    ns_list, nt_list = list(), list()
    for s, t in zip(src_o, tgt_o):
        ns, nt = np.linalg.norm(s), np.linalg.norm(t)
        ns_list.append(ns)
        nt_list.append(nt)
    ns_list=np.asarray(ns_list)
    nt_list=np.asarray(nt_list)

    dvd = nt_list/ns_list
    mean_sc = np.mean(dvd)

    src_o = src_o * mean_sc
    Z = np.matmul(src_o.transpose(), tgt_o)
    u, s, vh = np.linalg.svd(Z)
    M = np.matmul(u[:, :vh.shape[0]], vh)

    '''
    o_inv = np.linalg.pinv(src_o)
    M = np.matmul(o_inv, tgt_o)
    mean_sc = 1.0
    '''
    mean_sc = 1.0

    src_repr_names = list(src_repr.keys())
    src_w_name = src_repr_names[0]
    src_w = src_repr[src_w_name]
    src_w = src_w[:, :10]
    ali_src_w = np.matmul(M.transpose(), src_w * mean_sc)

    tgt_repr_names = list(tgt_repr.keys())
    tgt_w_name = tgt_repr_names[0]
    tgt_w = tgt_repr[tgt_w_name]
    tgt_w = tgt_w[:, :10]

    diff_w = tgt_w - ali_src_w

    feats = ali_src_w.flatten()
    feats = np.expand_dims(feats,axis=0)

    return feats



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




    model = torch.load(md_path)
    model.eval()
    label_tensor = torch.ones(len(good_inputs),dtype=torch.int64)
    input_tensor = torch.from_numpy(good_inputs)
    input_variable = Variable(input_tensor, requires_grad=True)
    logits = model(input_variable)
    preds = torch.argmax(logits,axis=-1)
    loss = F.cross_entropy(logits, label_tensor)
    loss.backward()

    fet = input_variable.grad.numpy()
    fet = np.average(fet, axis=0)
    fet = np.expand_dims(fet,0)
    return fet




    layer_name = list(reference.keys())[-4]
    ref_weight = reference[layer_name]
    layer_name = list(model.keys())[-4]
    w_ori = model[layer_name]
    weight = np.copy(w_ori)

    weight = align_weight(weight, ref_weight)

    n = weight.shape[0]
    # s_list = list()
    for i in range(n):
        a = weight[i]
        # s_list.append(np.sum(a))
        # a -= np.mean(a)
        # a /= np.std(a)
        # weight[i] = a
    # order = np.argsort(s_list)

    feat = list()
    # w = weight[order]
    # w = weight
    # for i in range(n):
    #    # w[i] /= np.linalg.norm(w[i])
    #    feat.append(ext_quantiles(w[i], bins=16))
    feat.append(ext_quantiles(weight[i].flatten(), bins=128))
    feat.append(ext_quantiles(np.abs(weight[i]).flatten(), bins=128))
    feat = np.concatenate(feat, axis=0)
    feat = np.expand_dims(feat,0)
    return feat


def kfold_validation(k_fold, dataset, train_fn, test_fn, configs):
    kfold = KFold(n_splits=k_fold, shuffle=True)

    labels = [data[1] for data in dataset]

    probs = np.zeros(len(dataset))
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info(f"FOLD {fold}")
        logging.info('-'*40)

        train_dataset = [dataset[i] for i in train_ids]
        test_dataset = [dataset[i] for i in test_ids]

        model, train_rst = train_fn(train_dataset, configs)
        test_rst = test_fn(model, test_dataset, configs)

        _probs = test_rst['probs']
        for k, i in enumerate(test_ids):
            probs[i] = _probs[k]

    auc = roc_auc_score(labels, probs)
    logging.info(f"Cross Validation AUC: {auc:3f}")

    model, train_rst = train_fn(dataset, configs)

    rst_dict = {
        'probs': probs,
        'labs': labels,
        'auc': auc,
        'model': model,
    }

    return rst_dict


def train_fn(dataset, configs):
    X = [data[0] for data in dataset]
    y = [data[1] for data in dataset]
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    clf = SVC(probability=True)
    # clf = lgb.LGBMClassifier(max_depth=2)
    clf.fit(X,y)

    preds = clf.predict(X)
    train_acc = np.sum(preds == y)/ len(y)
    logging.info("Train ACC: {:.3f}%".format(train_acc*100))

    train_rst = {
        'train_acc': train_acc
    }

    return clf, train_rst


def test_fn(model, dataset, configs):
    X = [data[0] for data in dataset]
    y = [data[1] for data in dataset]
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    test_acc = np.sum(preds == y)/len(y)
    logging.info(f"Test ACC: {test_acc*100:.3f}%")

    test_rst = {
        'acc': test_acc,
        'probs': probs[:,1],
        'labs': y,
    }
    return test_rst


def correlation_select(dataset, num_feats=100):
    X, y = list(), list()
    for data, label in dataset:
        X.append(data)
        y.append(label)
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    corr_list = list()
    for j in range(X.shape[1]):
        rst = np.corrcoef(X[:, j], y)[0,1]
        corr_list.append(rst)

    order = np.argsort(np.abs(corr_list))
    order = np.flip(order)

    for o in order:
        print(o, corr_list[o])

    return order[:num_feats]


def trim_dataset_according_id(dataset, feat_id):
    X, y = list(), list()
    for data, label in dataset:
        X.append(data)
        y.append(label)
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    tX = X[:, feat_id]

    new_dataset = list()
    for i in range(len(y)):
        new_dataset.append([tX[i:i+1,:], y[i]])

    return new_dataset



class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round
        self.number_features = metaparameters["train_number_features"]

    def write_metaparameters(self):
        metaparameters = {
            "train_number_features": self.number_features,
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        #for random_seed in np.random.randint(1000, 9999, 10):
        #    self.weight_table_params["random_seed"] = random_seed
        #    self.manual_configure(models_dirpath)
        self.number_features = 100

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict, model_dirpath_dict = load_models_dirpath(model_path_list)

        #==================gather clean inputs============================
        clean_dataset, poison_dataset = read_all_examples(model_dir=models_dirpath, out_folder=self.learned_parameters_dirpath, scaler_path=self.scale_parameters_filepath)

        path = join(self.learned_parameters_dirpath, "clean_data.pkl")
        with open(path,'rb') as fh:
            clean_dataset = pickle.load(fh)
        clean_inputs, clean_labels = clean_dataset
        good_ids = clean_labels==0
        good_inputs = clean_inputs[good_ids]


        #==================store reference model============================
        for p in model_path_list:
            if p.endswith('id-00000027'): #100% acc on clean examples
                break
        model_filepath = os.path.join(p,'model.pt')
        ref_feats = feature_extraction(model_filepath, good_inputs, model_repr=None, model_class=None, ref_feats=None)

        path = join(self.learned_parameters_dirpath, "reference_model.pkl")
        logging.info(f"Writing reference model to {path}")
        with open(path, 'wb') as fp:
            pickle.dump(ref_feats, fp)


        dataset = list()
        for model_class, model_list in model_repr_dict.items():
            for model_repr, label, md_path in zip(model_list, model_ground_truth_dict[model_class], model_dirpath_dict[model_class]):

                print(label, md_path)

                model_filepath = os.path.join(md_path, 'model.pt')
                aligned_feats = feature_extraction(model_filepath, good_inputs, model_repr=model_repr, model_class=model_class, ref_feats=ref_feats)

                dataset.append([aligned_feats, label])

        feat_id = correlation_select(dataset, num_feats=self.number_features)

        path = join(self.learned_parameters_dirpath, "feat_id.pkl")
        logging.info(f"Writing feat id to {path}")
        with open(path, 'wb') as fp:
            pickle.dump(feat_id, fp)

        dataset = trim_dataset_according_id(dataset, feat_id)

        kfold_validation(k_fold=4, dataset=dataset, train_fn=train_fn, test_fn=test_fn, configs=None)

        model, _ = train_fn(dataset, configs=None)


        '''
        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        # Flatten models
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)

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
                if X is None:
                    X = model_feats
                    continue

                X = np.vstack((X, model_feats)) * self.model_skew["__all__"]
        '''

        #logging.info("Training RandomForestRegressor model...")
        #model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        #model.fit(X, y)

        logging.info(f"Saving classify model {self.model_filepath}")
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

        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()

                pred = torch.argmax(model(feature_vector).detach()).item()

                ground_tuth_filepath = examples_dir_entry.path + ".json"

                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()

                print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        '''
        #with open(self.model_layer_map_filepath, "rb") as fp:
        #    model_layer_map = pickle.load(fp)

        # List all available model and limit to the number provided
        model_path_list = sorted(
            [
                join(round_training_dataset_dirpath, 'models', model)
                for model in listdir(join(round_training_dataset_dirpath, 'models'))
            ]
        )
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, _, _= load_models_dirpath(model_path_list)
        logging.info("Loaded models. Flattenning...")
        '''

        path = join(self.learned_parameters_dirpath, "clean_data.pkl")
        with open(path,'rb') as fh:
            clean_dataset = pickle.load(fh)
        clean_inputs, clean_labels = clean_dataset
        good_ids = clean_labels==0
        good_inputs = clean_inputs[good_ids]

        path = join(self.learned_parameters_dirpath, "reference_model.pkl")
        logging.info(f"Loading reference model from {path}")
        with open(path, 'rb') as fp:
            ref_feats = pickle.load(fp)

        path = join(self.learned_parameters_dirpath, "feat_id.pkl")
        logging.info(f"loading feat id from {path}")
        with open(path, 'rb') as fp:
            feat_id = pickle.load(fp)

        '''

        with open(self.models_padding_dict_filepath, "rb") as fp:
            models_padding_dict = pickle.load(fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        # Flatten model
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)
        '''

        aligned_feats = feature_extraction(model_filepath, good_inputs, model_repr=None, model_class=None, ref_feats=ref_feats)
        X = aligned_feats[:, feat_id]
        # model, model_repr, model_class = load_model(model_filepath)
        # model_repr = pad_model(model_repr, model_class, models_padding_dict)
        # flat_model = flatten_model(model_repr, model_layer_map[model_class])



        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        # self.inference_on_example_data(model, examples_dirpath)

        '''
        X = (
            use_feature_reduction_algorithm(layer_transform[model_class], flat_model)
            * self.model_skew["__all__"]
        )
        '''

        print(self.model_filepath)
        with open(self.model_filepath, "rb") as fp:
            clf = pickle.load(fp)

        probs = clf.predict_proba(X)[0]
        probability = str(probs[1])
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
