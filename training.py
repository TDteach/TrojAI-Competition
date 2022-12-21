import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from read_meta import read_csv, summaries_meta, select_by_conditions
from test import kfold_validation

from reversion import RevisionDetector
from test import kfold_validation

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

import pickle

sys.path.insert(0, '..')

HOME = os.environ['HOME']
ROOT = os.path.join(HOME, 'data/tdc_data')
FINAL_ROUND_FOLDER = os.path.join(ROOT, 'detection/final_round_test')


def ext_quantiles(a, bins=128, normalize=False):
    qs = [i/bins for i in range(bins)]
    if normalize:
        if len(a) == 1:
            s = 1.0
        else:
            s = np.std(a)
        m = np.mean(a)
        aa = (a-m)/s
        rst = np.quantile(aa, qs, interpolation='nearest')

        # rst = np.concatenate([rst, np.asarray([m, s])], axis=0)
    else:
        # rst = np.quantile(a, qs, method='nearest')
        rst = np.quantile(a, qs, interpolation='nearest')
    return rst


def ext_features_of_array(a):
    flatted_a = a.flatten()
    features_a = [ext_quantiles(flatted_a), ext_quantiles(np.abs(flatted_a))]
    features = np.concatenate(features_a, axis=0)
    return features


def ext_features_of_weight(w):
    fw = w.flatten()
    if len(fw) == 1:
        s = 1.0
    else:
        s = np.std(fw)
    m = np.mean(fw)
    w = (w-m)/s

    w_shape = w.shape
    if len(w_shape) <= 2:
        return np.asarray([ext_features_of_array(w)])

    tail = 1
    for z in w_shape[2:]:
        tail *= z
    new_shape = [w_shape[0], w_shape[1], tail]
    new_w = np.reshape(w, new_shape)

    fet_list = list()
    for c in range(tail):
        fet_list.append(ext_features_of_array(new_w[:, :, c]))

    features = np.asarray(fet_list)
    return features


def ext_features_of_model(model, weights_st=None, weights_ed=None, mode='silent'):
    weights_list = list(model.named_parameters())

    if weights_st is None: weights_st = 0
    if weights_ed is None: weights_ed = len(weights_list)

    k = 0
    fet_list = list()
    for i, (name, w) in enumerate(weights_list):
        if i < weights_st: continue

        fet_list.append(ext_features_of_weight(w.detach().cpu().numpy()))
        nk = k+len(fet_list[-1])
        if mode=='show':
            print(i, k,'-',nk, name, w.shape)
        k = nk
    features = np.concatenate(fet_list, axis=0)

    return features


def get_training_features(output_dir, model_dir, configs, save_out=None):
    pre, fo = os.path.split(model_dir)
    meta_path = os.path.join(pre, 'METADATA.csv')
    meta_csv = read_csv(meta_path)
    meta_summary = summaries_meta(meta_csv, max_uniques=1000)

    weights_st = configs['weights_st']
    md_archi = configs['model_architecture']
    rows, model_ids = select_by_conditions(meta_csv, conds={'model_architecture':md_archi}, meta_summary=meta_summary)

    labels, features = list(), list()
    for row, mid in zip(rows, model_ids):
        print(mid)
        if row['poisoned'] == 'True':
            lb = 1
        else:
            lb = 0

        model_path = os.path.join(model_dir, f'id-{mid:08d}', 'model.pt')

        model = torch.load(model_path)
        model.eval()

        # fet = ext_features_of_model(model, weights_st=weights_st, mode='show')
        fet = ext_features_of_model(model, weights_st=weights_st, mode='silent')

        labels.append(lb)
        features.append(fet)

    labels = np.asarray(labels)
    features = np.asarray(features)

    if save_out is not None:
        rst = {'labels':labels, 'features':features}
        with open(save_out, 'wb') as f:
            pickle.dump(rst, f)

    return labels, features


from inception import InceptionBlock
class MNIST_Detection_Network(nn.Module):
    def __init__(self, in_channels=32, n_filters=32, out_channels=2):
        super().__init__()
        self.n_filters = n_filters
        self.embedding = self.get_embedding(in_channels, n_filters)
        self.classifier = nn.Linear(in_features=4 * n_filters, out_features=out_channels)

    def get_embedding(self, in_channels, n_filters):
        embedding = nn.Sequential(
            InceptionBlock(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=n_filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=n_filters * 4,
                n_filters=n_filters,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=n_filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        return embedding

    def forward(self, x):
        emb = self.embedding(x)
        y = self.classifier(emb.view(-1, self.n_filters*4))
        return y



def evaluate_v2(model, loader, configs):
    all_inputs = list()
    all_labels = list()
    for inputs, labels in loader:
        all_inputs.append(inputs.numpy())
        all_labels.append(labels.numpy())

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    probs_list = list()
    num_channels = all_inputs.shape[1]
    for ch in range(num_channels):
        fet = all_inputs[:, ch, :]
        lab = all_labels

        clf = model[ch]
        probs = clf.predict_proba(fet)[:, 1]

        probs_list.append(probs)

    pred_scores = np.transpose(np.asarray(probs_list))

    rf_clf = model['rf_clf']
    preds = rf_clf.predict(pred_scores)
    probs = rf_clf.predict_proba(pred_scores)
    test_acc = np.sum(preds == all_labels) / len(all_labels)
    print(f'test acc: {test_acc:.3f}')

    test_rst = {
        'acc': test_acc,
        'probs': probs[:,1],
        'labs': all_labels,
    }
    return test_rst



def train_detection_model_v2(loader, configs):
    all_inputs = list()
    all_labels = list()
    for inputs, labels in loader:
        all_inputs.append(inputs.numpy())
        all_labels.append(labels.numpy())

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    rst_models = dict()
    probs_list = list()
    num_channels = all_inputs.shape[1]
    for ch in range(num_channels):
        fet = all_inputs[:, ch, :]
        lab = all_labels

        sc = np.nan
        while np.isnan(sc):
            # clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
            clf = SVC(kernel='linear', probability=True)
            clf.fit(fet, lab)
            probs = clf.predict_proba(fet)[:, 1]
            sc = np.corrcoef(probs, lab)[0, 1]

        probs_list.append(probs)
        rst_models[ch] = clf

    pred_scores = np.transpose(np.asarray(probs_list))

    # rf_clf = LGBMClassifier()
    rf_clf = SVC(kernel='linear', probability=True)
    rf_clf.fit(pred_scores, all_labels)
    rst_models['rf_clf'] = rf_clf
    preds = rf_clf.predict(pred_scores)
    train_acc = np.sum(preds == all_labels) / len(all_labels)
    print(f'train acc: {train_acc:.3f}')

    train_rst = {
        'train_acc': train_acc
    }

    return rst_models, train_rst



def evaluate(model, loader, configs):
    probs_list = list()
    acc, cnt = 0, 0
    for inputs, labels in loader:
        inputs = inputs.float().cuda()
        labels = labels.cuda()
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1)

        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs[:,1].detach().cpu().numpy())
        acc += torch.sum(preds.eq(labels)).item()
        cnt += len(labels)

    acc = acc/cnt
    print(f'Evaluation results: acc={acc:.3f}')

    if len(probs_list) == 1:
        probs = probs_list[0]
    else:
        probs = np.concatenate(probs_list, axis=0)
    test_rst = {
        'acc': acc,
        'probs': probs,
    }
    return test_rst



def train_detection_model(loader, feature_dim, configs):
    epochs = configs['detection_train_epochs']

    model = MNIST_Detection_Network(in_channels=feature_dim, n_filters=32)
    model = model.cuda()

    lr = configs['detection_train_lr']
    wd = configs['detection_train_weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader)*epochs)

    loss_ema = np.inf

    loss_list = list()
    pbar = tqdm(range(epochs))
    for epoch in pbar:

        for inputs, labels in loader:
            inputs = inputs.float().cuda()
            labels = labels.cuda()

            logits = model(inputs)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05

            pbar.set_description(f'{epoch}: Loss {loss:.3f}')
            loss_list.append(loss.item())

    train_rst = {
        'loss_list': loss_list
    }
    return model, train_rst


def svm_for_correlation(features, labels):
    sc_list = list()
    ch = features.shape[1]
    for i in range(ch):
        fet = features[:, i, :]
        lab = labels

        # '''
        dim = fet.shape[-1]
        max_sc = -np.inf
        for j in range(dim):
            x = fet[:, j]
            sc = np.corrcoef(x, lab)[0, 1]
            if np.isnan(sc):
                sc = 0
            sc = np.abs(sc)
            max_sc = max(sc, max_sc)
        sc_list.append(max_sc)
        # '''

        '''
        sc = np.nan
        while np.isnan(sc):
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
            a = clf.fit(fet, lab)
            probs = clf.predict_proba(fet)[:, 1]

            sc = np.corrcoef(probs, lab)[0, 1]

        sc_list.append(np.abs(sc))
        # '''

        # print(i, sc, np.min(probs), np.max(probs))


    order = np.argsort(sc_list)
    for o in order:
        print(o, sc_list[o])
    best_order = order[-48:]
    best_order.sort()

    print(np.mean([sc_list[o] for o in best_order]))
    print(best_order)

    best_features = features[:, best_order, :]

    return best_order, best_features


def learn(output_dir, model_dir, configs, save_out=None):

    all_probs = list()
    all_labs = list()
    model_types = configs['model_types']
    for ty in model_types:
        # ty = 'ResNet50'
        print(ty)
        saveout_name = ty
        saveout_path = f'training_features_{saveout_name}_normalized_128*2.pkl'

        _configs = configs[ty]
        # labels, features = get_training_features(output_dir, model_dir, _configs, save_out=saveout_path)

        print(_configs)

        with open(saveout_path,'rb') as f:
            data = pickle.load(f)
        labels, features = data['labels'], data['features']

        p = f'{ty}_best_roder.npy'
        p = os.path.join(output_dir,p)
        print(p)

        '''
        best_order, best_features = svm_for_correlation(features, labels)
        with open(p, 'wb') as f:
            np.save(f, best_order)
        # exit(0)
        # '''

        with open(p, 'rb') as f:
            best_order = np.load(f)
        best_features = features[:, best_order, :]

        features = best_features

        labels, features = torch.from_numpy(labels), torch.from_numpy(features)
        dataset = torch.utils.data.TensorDataset(features, labels)

        print(features.shape)

        rst_dict = kfold_validation(k_fold=4, dataset=dataset, train_fn=train_detection_model_v2, test_fn=evaluate_v2, configs=_configs)
        all_probs.append(rst_dict['probs'])
        all_labs.append(rst_dict['labs'])

        batch_size = _configs['detection_train_batch_size']
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model, train_rst = train_detection_model_v2(train_loader, configs)
        test_rst = evaluate_v2(model, train_loader, configs)

        model['best_order'] = best_order

        output_path = os.path.join(output_dir, saveout_name+'.pd')
        # torch.save(model.state_dict(), output_path)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)


    all_fets = list()
    all_tgts = list()
    i = 0
    for ty, probs, labs in zip(model_types, all_probs, all_labs):
        for p, l in zip(probs, labs):
            w = np.zeros(4,dtype=np.float32)
            w[i] = 1
            w[3] = p
            all_fets.append(w)
            all_tgts.append(l)
        i += 1
    all_fets = np.asarray(all_fets)
    all_tgts = np.asarray(all_tgts)

    # global_clf = SVC(kernel='linear', probability=True)
    global_clf = SVC(probability=True)
    global_clf.fit(all_fets, all_tgts)
    preds = global_clf.predict(all_fets)
    train_acc = np.sum(preds == all_tgts) / len(all_tgts)
    print('final train acc:', train_acc)

    output_path = os.path.join(output_dir, 'global.pd')
    with open(output_path, 'wb') as f:
        pickle.dump(global_clf, f)




if __name__ == '__main__':
    folder = FINAL_ROUND_FOLDER
    fns = os.listdir(folder)
    fns.sort()

    model_paths = list()
    for fo in fns:
        model_paths.append(os.path.join(folder, fo, 'model.pt'))

    scores = detection_by_weight_analysis(model_paths)
    # with open('init_scores.npy', 'rb') as f:
    #     scores = np.load(f)

    # '''
    adjusted_scores = list()
    RD = RevisionDetector()
    for md_path, sc in zip(model_paths, scores):
        if 0.8 > sc and sc > 0.6:
            print(md_path)
            rst_dict = RD.detect(md_path)
            print(rst_dict)
            asr = rst_dict['asr'] / 100.0
            if asr > 0.97:
                adjusted_scores.append(asr)
            else:
                adjusted_scores.append(sc)
        else:
            adjusted_scores.append(sc)
    scores = np.asarray(adjusted_scores)
    # '''

    sub_folder = 'my_submission'
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    with open(os.path.join(sub_folder, 'predictions.npy'), 'wb') as f:
        np.save(f, scores)


