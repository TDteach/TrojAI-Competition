import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
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



    a = src_ins[-1]
    b = tgt_ins[-1]

    return CKA(a,b)


    exit(0)

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


    exit(0)

    src_w = src_repr[src_repr_names[0*2]]
    tgt_w = tgt_repr[tgt_repr_names[0*2]]
    src_b = src_repr[src_repr_names[0*2+1]]
    tgt_b = tgt_repr[tgt_repr_names[0*2+1]]

    n = len(src_w)
    Z = np.corrcoef(src_w, tgt_w)
    Z -= np.eye(2*n, dtype=int)
    W = Z[n:, :n]
    print(W[0,:10])
    V = Z[:n, n:]
    print(V[:10, 0])
    aW = np.max(W, axis=1)
    agW = np.argmax(W, axis=1)
    aV = np.max(V, axis=1)
    agV = np.argmax(V, axis=1)
    print(np.sum(agW==agV))
    exit(0)

    feats = aW.flatten()
    feats = np.expand_dims(feats,axis=0)

    return feats


    src_i, tgt_i = src_ins[-1], tgt_ins[-1]

    # '''
    Z = np.matmul(src_i.transpose(), tgt_i)
    u, s, vh = np.linalg.svd(Z)
    M = np.matmul(u[:, :vh.shape[0]], vh)
    # '''

    '''
    i_inv = np.linalg.pinv(src_i)
    M = np.matmul(i_inv, tgt_i)

    # '''

    ali_src_w = np.matmul(src_w, M.transpose())
    mat = ali_src_w

    '''
    tgt_repr_names = list(tgt_repr.keys())
    tgt_w_name = tgt_repr_names[0]
    tgt_w = tgt_repr[tgt_w_name]

    diff_w = tgt_w - ali_src_w
    '''

    feats = mat[1].flatten()
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



def linear_adjust(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    lr = 0.1
    alpha = 1.0
    beta = 0.0

    sc = X-0.5
    sigmoid_sc = 1.0 / (1.0 + np.exp(-sc))
    sigmoid_sc = np.minimum(1.0 - 1e-12, np.maximum(0.0 + 1e-12, sigmoid_sc))
    loss = -(Y * np.log(sigmoid_sc) + (1 - Y) * np.log(1 - sigmoid_sc))
    print('init loss:', np.mean(loss))

    patience = 50
    best_loss = None
    best_alpha = alpha
    best_beta = beta
    for step in range(500000):
        g_beta = sigmoid_sc - Y
        g_alpha = g_beta * X

        alpha -= lr * np.mean(g_alpha)
        beta -= lr * np.mean(g_beta)

        sc = X * alpha + beta
        sigmoid_sc = 1.0 / (1.0 + np.exp(-sc))
        sigmoid_sc = np.minimum(1.0 - 1e-12, np.maximum(0.0 + 1e-12, sigmoid_sc))
        loss = -(Y * np.log(sigmoid_sc) + (1 - Y) * np.log(1 - sigmoid_sc))
        mean_loss = np.mean(loss)

        if best_loss is not None and mean_loss > best_loss-1e-9:
            patience -= 1
            if patience <= 0:
                break
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            best_alpha = alpha
            best_beta = beta

    print('loss:', best_loss)
    print(best_alpha, best_beta)
    return {'alpha': best_alpha, 'beta': best_beta}



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


def train_fn(dataset, configs):
    X = [data[0] for data in dataset]
    y = [data[1] for data in dataset]
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    if 'GBC_parameters' in configs:
        params = configs['GBC_parameters']
        clf = GradientBoostingClassifier(warm_start=True, **params)
    else:
        clf = GradientBoostingClassifier(learning_rate=0.1, warm_start=True, n_estimators=1000, tol=1e-7, max_features=24)

    # clf = SVC(probability=True)
    # clf = SVC(probability=True, kernel='poly') # for assemble matrix
    # clf = GradientBoostingClassifier(learning_rate=0.1, warm_start=True, n_estimators=1000, tol=1e-7, max_features=24)
    # clf = ExtraTreesClassifier(n_estimators=500, criterion='gini', warm_start=True, max_features=32)
    # clf = HistGradientBoostingClassifier(max_iter=1000, warm_start=True, loss='log_loss', l2_regularization=1)
    # clf = RandomForestClassifier(n_estimators=1000)
    # clf = lgb.LGBMClassifier(n_estimators=500)

    clf.fit(X,y)
    preds = clf.predict(X)
    train_acc = np.sum(preds == y)/ len(y)
    logging.info("Train ACC: {:.3f}%".format(train_acc*100))

    fimp = clf.feature_importances_

    train_rst = {
        'train_acc': train_acc,
        'fet_importance': fimp
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

    # order = np.argsort(np.abs(corr_list))
    order = np.argsort(corr_list)
    # order = np.flip(order)

    for o in order[:num_feats]:
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


def transform_X(X, scaler_path):
    scaler = StandardScaler()
    scale_params = np.load(scaler_path)
    scaler.mean_ = scale_params[0]
    scaler.scale_ = scale_params[1]

    new_X = list()
    for x in X:
        xx = x.reshape(1,-1)
        txx = scaler.transform(xx)
        new_X.append(txx.astype(np.float32))
    new_X = np.concatenate(new_X, axis=0)
    return new_X


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

        path = join(self.learned_parameters_dirpath, "reference_model.pkl")
        logging.info(f"Loading reference model from {path}")
        with open(path, 'rb') as fp:
            ref_feats = pickle.load(fp)

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

        exit(0)


        from vizier.service import clients
        from vizier.service import pyvizier as VZ

        problem = VZ.ProblemStatement()
        problem.search_space.root.add_categorical_param('loss', ['log_loss','deviance','exponential'])
        problem.search_space.root.add_int_param('n_estimators', 10, 1000)
        problem.search_space.root.add_float_param('subsample', 0.1, 1.0)
        problem.search_space.root.add_categorical_param('criterion', ['friedman_mse','squared_error'])
        problem.search_space.root.add_int_param('max_depth', 2, 20)
        problem.search_space.root.add_int_param('max_features', 2, 1000)
        problem.search_space.root.add_float_param('tol', 1e-6, 1.0)
        problem.metric_information.append(VZ.MetricInformation('AUC', goal=VZ.ObjectiveMetricGoal.MAXIMIZE))

        study_config = VZ.StudyConfig.from_problem(problem)
        study_config.algorithm = 'GAUSSIAN_PROCESS_BANDIT'

        study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')
        for i in range(10):
            suggestions = study.suggest(count=1)
            for suggestion in suggestions:
                params = suggestion.parameters
                params['n_estimators'] = int(params['n_estimators'])
                params['max_depth'] = int(params['max_depth'])
                params['max_features'] = int(params['max_features'])
                configs={'GBC_parameters': params}

                auc = params['tol']
                # kfold_rst = kfold_validation(k_fold=4, dataset=dataset, train_fn=train_fn, test_fn=test_fn, configs=configs)
                # auc = kfold_rst['auc']

                print(f'{i}: AUC={auc:.4f}')

                final_measurement = VZ.Measurement({'AUC': auc})
                suggestion.complete(final_measurement)

                # model = kfold_rst['model']

        for optimal_trial in study.optimal_trials():
            optimal_trial = optimal_trial.materialize()
            print('Optimal Trial Suggestion and Objective:')
            print(optimal_trial.parameters)
            print(optimal_trial.final_measurement)



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
        clean_dataset, poison_dataset = read_all_examples(model_dir=models_dirpath, out_folder=self.learned_parameters_dirpath)

        path = join(self.learned_parameters_dirpath, "clean_data.pkl")
        logging.info(f"Writing clean data to {path}")
        with open(path,'rb') as fh:
            clean_dataset = pickle.load(fh)
        clean_inputs, clean_labels = clean_dataset
        good_ids = clean_labels==0
        bad_ids = clean_labels==1
        good_inputs = clean_inputs[good_ids]
        bad_inputs = clean_inputs[bad_ids]
        #order_inputs = np.concatenate([good_inputs, bad_inputs], axis=0)
        #good_inputs = transform_X(order_inputs, self.scale_parameters_filepath)
        good_inputs = transform_X(bad_inputs, self.scale_parameters_filepath)

        #==================store reference model============================
        for p in model_path_list:
            if p.endswith('id-00000026'): #100% acc on clean examples
                break
        model_filepath = os.path.join(p,'model.pt')
        ref_feats = feature_extraction(model_filepath, good_inputs, model_repr=None, model_class=None, ref_feats=None)

        path = join(self.learned_parameters_dirpath, "reference_model.pkl")
        logging.info(f"Writing reference model to {path}")
        with open(path, 'wb') as fp:
            pickle.dump(ref_feats, fp)

        #with open('train_dataset.pkl','rb') as fp:
        #    dataset = pickle.load(fp)

        #'''
        dataset = list()
        for model_class, model_list in model_repr_dict.items():
            print(model_class)

            if model_class.endswith('r'):
                vv =1
                cc = int(model_class[-2])
            elif model_class.endswith('s'):
                vv =2
                cc = int(model_class[-2])
            else:
                vv=0
                cc = int(model_class[-1])

            print(cc, vv)

            # if not model_class.startswith('Net5'): continue

            for model_repr, label, md_path in zip(model_list, model_ground_truth_dict[model_class], model_dirpath_dict[model_class]):

                # if label == 1 : continue

                print(label, md_path)

                model_filepath = os.path.join(md_path, 'model.pt')
                aligned_feats = feature_extraction(model_filepath, good_inputs, model_repr=model_repr, model_class=model_class, ref_feats=ref_feats)

                dataset.append([aligned_feats, label, cc, vv])

        with open('train_dataset.pkl','wb') as fp:
            pickle.dump(dataset, fp)

        # '''

        print('quit')
        exit(0)


        '''
        feat_id = correlation_select(dataset, num_feats=self.number_features)

        path = join(self.learned_parameters_dirpath, "feat_id.pkl")
        logging.info(f"Writing feat id to {path}")
        with open(path, 'wb') as fp:
            pickle.dump(feat_id, fp)

        dataset = trim_dataset_according_id(dataset, feat_id)
        # '''

        kfold_rst = kfold_validation(k_fold=4, dataset=dataset, train_fn=train_fn, test_fn=test_fn, configs=None)
        model = kfold_rst['model']

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
        logging.info(f"Writing clean data to {path}")
        with open(path,'rb') as fh:
            clean_dataset = pickle.load(fh)
        clean_inputs, clean_labels = clean_dataset
        good_ids = clean_labels==0
        bad_ids = clean_labels==1
        good_inputs = clean_inputs[good_ids]
        bad_inputs = clean_inputs[bad_ids]
        order_inputs = np.concatenate([good_inputs, bad_inputs], axis=0)
        good_inputs = transform_X(order_inputs, self.scale_parameters_filepath)


        path = join(self.learned_parameters_dirpath, "reference_model.pkl")
        logging.info(f"Loading reference model from {path}")
        with open(path, 'rb') as fp:
            ref_feats = pickle.load(fp)

        feat_id = None

        '''
        path = join(self.learned_parameters_dirpath, "feat_id.pkl")
        logging.info(f"loading feat id from {path}")
        with open(path, 'rb') as fp:
            feat_id = pickle.load(fp)

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
        if feat_id is not None:
            X = aligned_feats[:, feat_id]
        else:
            X = aligned_feats
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

        model_filepath = self.model_filepath
        model_filepath = os.path.join(self.learned_parameters_dirpath, 'automl_model.pkl')

        logging.info(f"loading model from {model_filepath}")
        with open(model_filepath, "rb") as fp:
            clf = pickle.load(fp)

        probs = clf.predict_proba(X)[0]
        # probability = str((probs[1]-0.5)*0.9+0.5)
        probability = str(probs[1])
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
