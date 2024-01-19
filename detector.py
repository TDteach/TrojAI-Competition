# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from collections import OrderedDict
import logging
import os
import json
import jsonpickle
import pickle
import numpy as np
import copy

from sklearn.ensemble import RandomForestRegressor

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torch.nn as nn
try:
    import torch_ac
except:
    pass

from torch import optim
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GroupKFold

import autosklearn
import autosklearn.classification
import autosklearn.metrics
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA

from typing import Optional

from tqdm import tqdm

from feature_selector import FE_layer, FE_global, select_features_from_As

# import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split




def ext_quantiles(a, bins=100):
    qs = [i/bins for i in range(bins)]
    return np.quantile(a, qs)

def ext_weight_matrix(w):
    ww = np.reshape(w, [len(w),-1])
    return 0


def features_extraction_based_map(model_repr, fet_map):
    fet = []
    keys = list(model_repr.keys())
    for nl,i in fet_map:
        z = model_repr[keys[nl]].flatten()
        fet.append(z[i])
    return np.asarray(fet)


def feature_extraction(model_class, model_repr):
    ip = None
    ln_list = list(model_repr.keys())

    if model_class=='BasicFCModel':
        for i in range(6,-1,-1):
            ln = ln_list[i]
            w = model_repr[ln]
            if len(w.shape) < 2:
                continue
            # print(i, ln)
            if ip is None:
                ip = np.copy(w)
            else:
                ip = np.matmul(ip, w)

            for r in ip:
                r /= np.linalg.norm(r)
        cp = None
        for i in [10, 8, 2, 0]:
            ln = ln_list[i]
            w = model_repr[ln]
            # print(i, ln)
            if cp is None:
                cp = np.copy(w)
            else:
                cp = np.matmul(cp, w)

            for r in cp:
                r /= np.linalg.norm(r)
        ip = np.concatenate([ip,cp], axis=0)


    elif model_class=='SimplifiedRLStarter':
        # for i in range(len(ln_list)):
            # print(i, ln_list[i], model_repr[ln_list[i]].shape)
        # exit(0)
        for i in range(10,5,-1):
            ln = ln_list[i]
            w = model_repr[ln]
            if len(w.shape) < 2:
                continue
            # print(i, ln)
            if ip is None:
                ip = np.copy(w)
            else:
                ip = np.matmul(ip, w)

            for r in ip:
                r /= np.linalg.norm(r)
        ip = np.expand_dims(ip, [2,3])

        # print(ip.shape)
        w = model_repr['image_conv.5.weight']
        z = F.conv_transpose2d(torch.from_numpy(ip), torch.from_numpy(w))
        # print(z.shape)
        ww = model_repr['image_conv.3.weight']
        v = F.conv_transpose2d(z, torch.from_numpy(ww))
        # print(v.shape)
        u = F.upsample(v, scale_factor=[2,2])
        www = model_repr['image_conv.0.weight']
        x = F.conv_transpose2d(u, torch.from_numpy(www))
        # print(x.shape)
        x = x.numpy()
        x = np.reshape(x,[3,-1])
        # print(x.shape)

        ip = x
        for r in ip:
            r /= np.linalg.norm(r)

        cp = None
        for i in [16, 14, 12]:
            ln = ln_list[i]
            w = model_repr[ln]
            # print(i, ln)
            if cp is None:
                cp = np.copy(w)
            else:
                cp = np.matmul(cp, w)
            for r in cp:
                r /= np.linalg.norm(r)

        cp = np.expand_dims(cp, [2,3])
        w = model_repr['image_conv.5.weight']
        z = F.conv_transpose2d(torch.from_numpy(cp), torch.from_numpy(w))
        ww = model_repr['image_conv.3.weight']
        v = F.conv_transpose2d(z, torch.from_numpy(ww))
        u = F.upsample(v, scale_factor=[2,2])
        www = model_repr['image_conv.0.weight']
        x = F.conv_transpose2d(u, torch.from_numpy(www))
        x = x.numpy()
        x = np.reshape(x,[1,-1])

        cp = x
        for r in cp:
            r /= np.linalg.norm(r)
        ip = np.concatenate([ip,cp], axis=0)

    # print(ip)
    return ip

def test_fn(model, dataset, configs):
    X = [data[0] for data in dataset]
    y = [data[1] for data in dataset]
    X = np.asarray(X)
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

def train_fn(dataset, configs):
        X = [data[0] for data in dataset]
        y = [data[1] for data in dataset]
        X = np.asarray(X)
        y = np.asarray(y)

        model = GradientBoostingClassifier(learning_rate=0.05, warm_start=True,  n_estimators=200, tol=1e-7, max_features=64)
        print(X.shape)
        print(y.shape)
        model.fit(X, y)
        preds = model.predict(X)
        train_acc = np.sum(preds == y) / len(y)
        logging.info("Train ACC: {:.3f}%".format(train_acc*100))

        fimp = model.feature_importances_

        train_rst = {
            'train_acc': train_acc,
            'fet_importance': fimp,
        }

        return model, train_rst

def kfold_validation(k_fold, dataset, train_fn, test_fn, configs):
        kfold = KFold(n_splits=k_fold, shuffle=True)

        labels = [data[1] for data in dataset]

        fimp = None
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

            if 'fet_importance' in train_rst:
                if fimp is None:
                    fimp = train_rst['fet_importance']
                else:
                    fimp += train_rst['fet_importance']

        if fimp is not None:
            forder = np.argsort(fimp)
            forder = np.flip(forder)
            for o in forder[:32]:
                print(o, fimp[o])
            print(np.min(forder))


        auc = roc_auc_score(labels, probs)
        ce_score = log_loss(labels, probs)
        logging.info(f"Cross Validation AUC: {auc:.6f}")
        logging.info(f"Cross Validation CE: {ce_score:.6f}")

        #probs = (probs-0.5)*0.9+0.5
        #auc = roc_auc_score(labels, probs)
        #ce_score = log_loss(labels, probs)
        #logging.info(f"Cross Validation AUC: {auc:.6f}")
        #logging.info(f"Cross Validation CE: {ce_score:.6f}")

        # adjust_rst = linear_adjust(probs, labels)

        model, train_rst = train_fn(dataset, configs)

        rst_dict = {
            'probs': probs,
            'labs': labels,
            'auc': auc,
            'model': model,
        }

        return rst_dict








class MLP(nn.Module):
    def __init__(self, din=10, dout=1, num_filters=16, depth=1):
        super(MLP, self).__init__()
        self.din=din
        self.dout=dout
        self.num_filters=num_filters
        self.depth = depth

        if depth == -1:
            self.features = nn.Linear(din, 1)
        elif depth == 0 :
            self.features = nn.Identity()
            num_filters = din
        else:
          self.features = nn.Sequential()

          for i in range(self.depth):
            if i == 0:
                self.features.add_module('linear%02d'%(i+1), nn.Linear(self.din, self.num_filters))
            else:
                self.features.add_module('linear%02d'%(i+1), nn.Linear(self.num_filters, self.num_filters))

            if i < self.depth-1:
                self.features.add_module('activation%02d'%(i+1), nn.LeakyReLU(inplace=True))
            else:
                pass
                # self.features.add_module('activation%02d'%(i+1), nn.Tanh())

        self.classifier = nn.Linear(num_filters, self.dout)


    def forward(self, x, return_embeds=False):
        _embeds = self.features(x)
        # embeds = torch.tanh(embeds)
        # if return_embeds:
            # return embeds
        if len(_embeds.shape) == 3:
            embeds, _ = torch.max(_embeds, dim=-1)
        else:
            embeds = _embeds
        _logits = self.classifier(embeds)
        if len(_logits.shape) == 3:
            logits, _ = torch.max(_logits, dim=1)
        else:
            logits = _logits
        return logits
    
    def init_weights(self, m):
        if type(m) == nn.Liner:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.00)

    def reset(self):
        self.features.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    def get_L1loss(self):
        loss = 0
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                # loss += torch.mean(layer.weight.abs())
                loss += torch.sum(layer.weight.abs())
                break
        return loss



def train_classifier_on_layers(repr_dicts, labels, selected_layers, n_fold=5):
    labels = np.asarray(labels)
    labels = np.expand_dims(labels, axis=1)

    data = []
    for repr_dict in repr_dicts:
        a = []
        for layer in selected_layers:
            weight = repr_dict[layer]
            if len(weight.shape) == 1:
                weight = np.expand_dims(weight, axis=0)
            for i in range(len(weight)):
                weight[i, :] /= np.linalg.norm(weight[i])
            if weight.shape[1] != 768: continue
            # print(weight.shape)
            a.append(weight)
        a = np.concatenate(a, axis=0)
        print(a.shape)
        data.append(a)
    data= np.asarray(data)

    return train_mlp(data, labels, n_fold, num_filters=6153, depth=-1)
    # return train_mlp(data, labels, n_fold)


def train_mlp(data, labels, n_fold=5, num_filters=8, depth=2, l1_weight=None):
    
    n = len(data)

    index = np.random.permutation(n)
    ntest = n//n_fold
    ntrain = n-ntest

    train_idx = index[:ntrain]
    test_idx = index[ntrain:]

    train_data = data[train_idx.astype(int)]
    train_label = labels[train_idx.astype(int)]
    test_data = data[test_idx.astype(int)]
    test_label = labels[test_idx.astype(int)]

    '''
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, SelectFromModel
    from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
    from sklearn.pipeline import Pipeline

    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    min_features_to_select = 32  # Minimum number of features to consider
    clf = LogisticRegression()
    cv = StratifiedKFold(5)

    labels = np.squeeze(labels, -1)
    rfecv = RFECV(
    estimator=clf,
    step=0.01,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=20,
    verbose=2,
    )
    rfecv.fit(data, labels)

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(rfecv.score(data, labels))

    exit(0)
    # '''



    d_data = data.shape[-1]
    # mlp = MLP(din=d_data, dout=1, num_filters=6153, depth=-1)
    mlp = MLP(din=d_data, dout=1, num_filters=num_filters, depth=depth)
    mlp = mlp.cuda().train()

    steps = 1000
    batch_size = 32

    optimizer = optim.NAdam(mlp.parameters(), lr=1e-5, betas=[0.8,0.95])
    # optimizer = optim.Adam(mlp.parameters(), lr=1e-5, betas=[0.8, 0.95])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    data_tensor = torch.from_numpy(train_data).cuda()
    label_tensor = torch.from_numpy(train_label).cuda()

    pbar = tqdm(range(steps))
    for _ in pbar:
        idx = torch.randperm(ntrain)[:batch_size]

        x = data_tensor[idx]
        y = label_tensor[idx]
        # y = label_tensor[idx, :] * 2 - 1

        logits_raw = mlp(x)
        loss = F.binary_cross_entropy_with_logits(logits_raw, y.float())
        # tmd = logits*y
        # loss = -torch.mean(tmd)

        # if l1_weight is not None:
        if False:
            l1_loss = mlp.get_L1loss()
            print(l1_loss.item())
            loss += l1_loss * l1_weight
        print(loss.item())
        # print(torch.sum(tmd > 0).item()/batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(loss.item())

    data_tensor = torch.from_numpy(test_data).cuda()
    label_tensor = torch.from_numpy(test_label).cuda()
    corrects = 0
    mlp = mlp.eval()
    with torch.no_grad():
        logits = mlp(data_tensor)
        preds = logits > 0
        corrects += torch.eq(preds, label_tensor).sum().item()
    print('test acc:', corrects/len(test_label))

    return mlp


def convert_repr_to_features(repr_dicts, layer_rst):
    logits_ary = []
    for key in layer_rst:
        selected_layers = layer_rst[key]['selected_layers']
        layer_mlp = layer_rst[key]['layer_mlp']

        data = []
        for repr_dict in repr_dicts:
            a = []
            for layer in selected_layers:
                weight = repr_dict[layer]
                if len(weight.shape) == 1:
                    weight = np.expand_dims(weight, axis=0)
                for i in range(len(weight)):
                    weight[i, :] /= np.linalg.norm(weight[i])
                if weight.shape[1] != 768: continue
                a.append(weight)
            a = np.concatenate(a, axis=0)
            data.append(a)
        data= np.asarray(data)
        print(key, data.shape)

        layer_mlp = layer_mlp.cuda()
        data = torch.from_numpy(data).cuda()
        with torch.no_grad():
            embeds = layer_mlp(data, return_embeds=True).detach().cpu().numpy()

        logits_ary.append(embeds)
        layer_mlp = layer_mlp.cpu()

    logits_ary = np.concatenate(logits_ary, axis=1)

    return logits_ary


def train_final_classifier(repr_dicts, labels, layer_rst, n_fold=5):
    labels = np.asarray(labels)
    labels = np.expand_dims(labels, axis=1)
    data = convert_repr_to_features(repr_dicts, layer_rst)
    print(data.shape)
    return train_mlp(data, labels, n_fold, num_filters=32, depth=2)









class MutualInfoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, input_features, random_state=None):
        # self.input_features = 100
        # self.f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
        self.input_features = input_features
        self.f_selector = None
        self.random_state = random_state

    def fit(self, X, y=None):
        self.f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
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


def select_top_cosine(xx, y, top=128):
    xx = xx.transpose().astype(np.float64)
    for i in range(xx.shape[0]):
        xx[i] /= np.linalg.norm(xx[i])
    z = np.matmul(xx, y)
    o = np.argsort(-np.abs(z))[:top]
    return z[o], o

def split_train_test_repr_dicts(repr_dicts, labels, train_portion=0.8):
    labels = np.asarray(labels)
    n0 = np.sum(labels==0)
    n1 = np.sum(labels==1)
    n = len(repr_dicts)
    assert n == n0+n1

    shuffled = np.arange(n)
    np.random.shuffle(shuffled)

    tr_n0 = int(n0*train_portion)
    tr_n1 = int(n1*train_portion)
    print(f'split train n: {tr_n0+tr_n1} n0: {tr_n0}, n1: {tr_n1}')
    print(f'split test n: {n-tr_n0-tr_n1} n0: {n0-tr_n0}, n1: {n1-tr_n1}')

    tr_idx, te_idx = [], []
    for i in shuffled:
        if labels[i] == 0:
            if tr_n0 > 0:
                tr_idx.append(i)
                tr_n0 -= 1
            else:
                te_idx.append(i)
        else:
            if tr_n1 > 0:
                tr_idx.append(i)
                tr_n1 -= 1
            else:
                te_idx.append(i)

    train_repr_dicts, train_labels = [], []
    for i in tr_idx:
        train_repr_dicts.append(repr_dicts[i])
        train_labels.append(labels[i])
    train_labels = np.asarray(train_labels)

    test_repr_dicts, test_labels = [], []
    for i in te_idx:
        test_repr_dicts.append(repr_dicts[i])
        test_labels.append(labels[i])
    test_labels = np.asarray(test_labels)

    return train_repr_dicts, train_labels, test_repr_dicts, test_labels


def test_model_for_repr_dicts(repr_dicts, labels, model, infer=False):
    clf_model = model['classifier']
    position = model['fet_selector']

    X_list = []
    for layer in position:
        As = list()
        for repr_dict in repr_dicts:
            As.append(repr_dict[layer])
        print(len(As))

        print(layer, repr_dict[layer].shape)
        fet_mat, _ = select_features_from_As(As,  position=position[layer])
        X_list.append(fet_mat)
    X_cat = np.concatenate(X_list, axis=1)

    if isinstance(clf_model, dict):
        selector, clf = clf_model['selector'], clf_model['clf']
    else:
        selector, clf = None, clf_model
    if selector is not None:
        X_cat = X_cat[:, selector]
    if infer:
        probs = clf.predict_proba(X_cat)[:, 1]
        return probs

    X_test, y_test = X_cat, labels

    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))

    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ce_score = log_loss(y_test, probs)
    print(f"AUC score: {auc:.6f}")
    print(f"CE score: {ce_score:.6f}")

    return auc, ce_score


def train_model_for_repr_dicts(repr_dicts, labels, test_repr_dicts=None, test_labels=None):

    def _train_final(X_train, y_train):
        params = {
            'n_estimators': 1000,
            'subsample': 0.6,
            'max_depth': 3,
            'learning_rate': 0.05,
            # 'verbose': 2,
        }
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        selector=None
        fimp = clf.feature_importances_
        if len(fimp) > 512:
            print('extract 512 features from', len(fimp))
            od = np.argsort(fimp)[-512:]
            selector = np.sort(od)
            X_train = X_train[:, selector]
            clf = GradientBoostingClassifier(**params)
            clf.fit(X_train, y_train)

        predictions = clf.predict(X_train)
        print(classification_report(y_train, predictions))

        return {
            'clf': clf,
            'selector': selector
        }
    
    def _train_global(X_list, labels, p_dict):
        X_cat = np.concatenate(X_list, axis=1)
        gX, g_position = FE_global(X_cat, labels, p_dict)

        # X_train, X_test, y_train, y_test = train_test_split(gX, labels, test_size=0.2, random_state=42)
        clf = _train_final(gX, labels)
        model = {
            'fet_selector': g_position,
            'classifier': clf,
        }
        return model

    X_list, p_dict = [], OrderedDict()

    layers = list(repr_dicts[0].keys())
    print(layers)

    best_auc = None
    best_score = None
    best_model = None
    do_k = 0
    for layer in tqdm(layers):
        if 'embeddings' in layer:
            continue
        if do_k > 20:
            break
        print(layer)
        As = list()
        for repr_dict in repr_dicts:
            As.append(repr_dict[layer])
        print(len(As))

        X, position = FE_layer(As, labels)
        X_list.append(X)
        p_dict[layer] = position
        # break

        if test_repr_dicts is not None and test_labels is not None:
            model = _train_global(X_list, labels, p_dict)
            auc, ce_score = test_model_for_repr_dicts(test_repr_dicts, test_labels, model)

            if best_auc is None or auc > best_auc or (auc == best_auc and ce_score < best_score):
                print(f'!!!!!! best_AUC update from {best_auc} to {auc}')
                print(f'!!!!!! best_score update from {best_score} to {ce_score}')
                best_auc = auc
                best_score = ce_score
                best_model = copy.deepcopy(model)
        do_k += 1

    if best_model is None:
        best_model = _train_global(X_list, labels, p_dict)
    return best_model



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
        # self.train_data_augment_factor = metaparameters["train_data_augment_factor"]
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
        for random_seed in np.random.randint(1000, 9999, 10):
            # self.weight_params["rso_seed"] = random_seed
            self.manual_configure(models_dirpath)
            break

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

        if True:
            model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)
            # print(list(model_repr_dict.keys())) #['RobertaForQuestionAnswering', 'MobileBertForQuestionAnswering']
            data = {
                'model_repr_dict': model_repr_dict,
                'model_ground_truth_dict': model_ground_truth_dict,
            }
            with open('all_data.pkl', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open('all_data.pkl', 'rb') as f:
                data = pickle.load(f)
            model_repr_dict = data['model_repr_dict']
            model_ground_truth_dict = data['model_ground_truth_dict']
        exit(0)

        num_features = self.input_features

        g_model = dict()

        with open('model.pkl', 'rb') as f:
            g_model = pickle.load(f)

        logging.info("Building RandomForest based on random features, with the provided mean and std.")
        # rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        for model_arch in model_repr_dict.keys():
            print(model_arch)
            if '104' in model_arch:
                continue
            if '200' in model_arch:
                continue
            # if '1114' in model_arch:
                # continue
            labels = []
            n = len(model_repr_dict[model_arch])
            for model_index in range(n):
                labels.append(model_ground_truth_dict[model_arch][model_index])
            labels = np.asarray(labels)

            repr_dicts = model_repr_dict[model_arch]

            train_repr_dicts, train_labels, test_repr_dicts, test_labels = split_train_test_repr_dicts(repr_dicts, labels, train_portion=0.8)
            model = train_model_for_repr_dicts(train_repr_dicts, train_labels, test_repr_dicts=test_repr_dicts, test_labels=test_labels)

            with open(f'{model_arch}.pkl', 'wb') as f:
                pickle.dump(model, f)

            with open(f'{model_arch}.pkl', 'rb') as f:
                model = pickle.load(f)

            test_model_for_repr_dicts(test_repr_dicts, test_labels, model)

            g_model[model_arch] = model

        with open('model.pkl', 'wb') as f:
            pickle.dump(g_model, f)

        self.write_metaparameters()
        logging.info("Configuration done!")



    def train_model_for_Xy(self, X, y, prefix=None, f_selector=None):
        from autosklearn.classification import AutoSklearnClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score 

        '''
        model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=2, random_state=0)
        scores = cross_val_score(model, X, y, cv=self.automl_num_folds, scoring='roc_auc')
        print(scores.shape)
        print(scores)

        print('AUC', np.average(scores))

        model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=2, random_state=0)
        model.fit(X,y)
        automl = model
        '''

        # '''
        automl = AutoSklearnClassifier(
            # include={"feature_preprocessor":["MutualInfoPreprocessing"]},
            metric=autosklearn.metrics.roc_auc,
            # resampling_strategy=resampling_strategy,
            resampling_strategy='cv',
            # resampling_strategy_arguments={'folds': self.train_data_augment_factor, 'groups': a},
            resampling_strategy_arguments={'folds': self.automl_num_folds},
            **self.automl_kwargs,
        )
        print('automl has been set up')
        automl.fit(X, y)

        print(automl.leaderboard(ensemble_only=False))
        # pprint(automl.show_models(), indent=4)
        print(automl.sprint_statistics())
        automl.refit(X, y)
        # '''

        model = {
            'f_selector': f_selector,
            'automl': automl,
        }

        logging.info("Saving model...")
        if prefix is not None:
            model_filepath = os.path.join(self.learned_parameters_dirpath, f'{prefix}_automl_model.pkl')
        else:
            model_filepath = os.path.join(self.learned_parameters_dirpath, 'automl_model.pkl')
        with open(model_filepath, 'wb') as fh:
            pickle.dump(model, fh)

        return model





    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        import gym
        from gym_minigrid.wrappers import ImgObsWrapper

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        model.to(device)
        model.eval()

        preprocess = torch_ac.format.default_preprocess_obss

        # Utilize open source minigrid environment model was trained on
        env_string_filepath = os.path.join(examples_dirpath, 'env-string.txt')
        with open(env_string_filepath) as env_string_file:
            env_string = env_string_file.readline().strip()
        logging.info('Evaluating on {}'.format(env_string))

        # Number of episodes to run
        episodes = 100

        env_perf = {}

        # Run episodes through an environment to collect what may be relevant information to trojan detection
        # Construct environment and put it inside a observation wrapper
        env = ImgObsWrapper(gym.make(env_string))
        obs, info = env.reset()
        obs = preprocess([obs], device=device)

        final_rewards = []
        with torch.no_grad():
            # Episode loop
            for _ in range(episodes):
                done = False
                # Use env observation to get action distribution
                dist, value = model(obs)
                # Per episode loop
                while not done:
                    # Sample from distribution to determine which action to take
                    action = dist.sample()
                    action = action.cpu().detach().numpy()
                    # Use action to step environment and get new observation
                    obs, reward, done, truncated, info = env.step(action)
                    # Preprocessing function to prepare observation from env to be given to the model
                    obs = preprocess([obs], device=device)
                    # Use env observation to get action distribution
                    dist, value = model(obs)

                # Collect episode performance data (just the last reward of the episode)
                final_rewards.append(reward)
                # Reset environment after episode and get initial observation
                obs, info = env.reset()
                obs = preprocess([obs], device=device)

        # Save final rewards
        env_perf['final_rewards'] = final_rewards

    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
            tokenizer_filepath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # load the model
        model, model_repr, model_class = load_model(model_filepath)

        # print(model)
        # exit(0)

        # Inferences on examples to demonstrate how it is done for a round
        # self.inference_on_example_data(model, examples_dirpath)

        # build a fake random feature vector for this model, in order to compute its probability of poisoning
        # rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        # X = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1, self.input_features))

        '''
        # load the RandomForest from the learned-params location
        model_filepath = os.path.join(self.learned_parameters_dirpath, f'model.pkl')
        with open(model_filepath, "rb") as fp:
            model = pickle.load(fp)

        # use the RandomForest to predict the trojan probability based on the feature vector X
        probability = model.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)
        print(probability)
        '''

        if True:
            print(model_class)
            model_filepath = os.path.join(self.learned_parameters_dirpath, 'model.pkl')
            with open(model_filepath, 'rb') as fh:
                model = pickle.load(fh)

            model = model[model_class]
            probs = test_model_for_repr_dicts([model_repr], None, model, infer=True)

            probability = probs[0]
            print(probability)


        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))




if __name__ == '__main__':
    mo_list = list()
    fn = ['roberta_104.pkl','roberta_200.pkl','mobilebert.pkl']
    for pt in fn:
        with open(pt,'rb') as fh:
            mo = pickle.load(fh)
        mo_list.append(mo)
    model = {
        'RobertaForQuestionAnswering_104': mo_list[0],
        'RobertaForQuestionAnswering_200': mo_list[1],
        'MobileBertForQuestionAnswering_1114': mo_list[2],
    }
    with open('model.pkl','wb') as fh:
        pickle.dump(model, fh)