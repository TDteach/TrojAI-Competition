import copy
import os
import pickle

import numpy as np

from utils import read_csv, dirpath_to_contest_phrase
from batch_run_trojai import contest_round_list, home
from example_trojan_detector import get_feature, global_hash_map

model_architecture = ['roberta-base', 'google/electra-small-discriminator', 'distilbert-base-cased']
# record_folder = 'record_results'
record_folder = 'scratch'

def prepare_data():
    gt_lb = dict()
    for contest_round in contest_round_list:
        folder_root = os.path.join(home, 'share', 'trojai', contest_round)
        gt_csv_path = os.path.join(folder_root, 'METADATA.csv')
        gt_csv = read_csv(gt_csv_path)

        for row in gt_csv:
            md_name = row['model_name']
            poisoned = row['poisoned']

            if poisoned == 'True':
                lb = 1
            else:
                lb = 0

            contest_phrase = dirpath_to_contest_phrase(contest_round)
            id_name = contest_round+'-'+md_name
            gt_lb[id_name] = {'lb':lb}

    fns = os.listdir(record_folder)
    for fn in fns:
        if not fn.endswith('.pkl'): continue
        id_name = fn.split('.')[0]
        if id_name not in gt_lb: continue
        gt_lb[id_name]['rd_path'] = os.path.join(record_folder, fn)

    del_list = list()
    for id_name in gt_lb:
        if 'rd_path' not in gt_lb[id_name]:
            del_list.append(id_name)
    for id_name in del_list:
        del gt_lb[id_name]

    for id_name in gt_lb:
        path = gt_lb[id_name]['rd_path']

        print(id_name)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        gt_lb[id_name]['raw'] = data
        feat = get_feature(data)
        gt_lb[id_name]['feature'] = feat

        print(id_name, gt_lb[id_name]['lb'], gt_lb[id_name]['feature'])


    return gt_lb


def linear_adjust(X, Y):
    lr = 1.0
    alpha = 1.0
    beta = 0.0

    sc = X
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
    # calc_auc(Y,sigmoid_sc)

    print(best_alpha, best_beta)

    return {'alpha': best_alpha, 'beta': best_beta}



def train_only_lr(gt_lb):
    X = np.asarray([gt_lb[k]['probs'][0] for k in gt_lb])
    Y = np.asarray([gt_lb[k]['lb'] for k in gt_lb])
    C = np.asarray([gt_lb[k]['probs'][1] for k in gt_lb])

    lr_param_dict = dict()
    from sklearn.metrics import roc_auc_score
    for t in range(3):
        indice = C == t
        _X = X[indice]
        _Y = Y[indice]
        for k in global_hash_map:
            if global_hash_map[k] == t:
                break
        ta = k
        auc = roc_auc_score(_Y, _X)
        print('='*20)
        print('task', ta, 'auc: %.4f' % (auc))
        best_lr_param = linear_adjust(_X, _Y)
        lr_param_dict[t] = best_lr_param

    adj_param = {'lr_param_dict': lr_param_dict, 'hash_map': global_hash_map}
    outpath = 'adj_lr_param.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(adj_param, f)
    print('dump to', outpath)


def train_rf(gt_lb):
    if gt_lb is not None:
        X = [gt_lb[k]['feature'] for k in gt_lb]
        Y = [gt_lb[k]['lb'] for k in gt_lb]
        X = np.asarray(X)
        Y = np.asarray(Y)

        out_data = {'X': X, 'Y': Y}
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(out_data, f)
        print('writing to train_data.pkl')

        print('X shape:', X.shape)
        print('Y shape:', Y.shape)


    from sklearn.model_selection import KFold
    # from mlxtend.classifier import StackingCVClassifier, StackingClassifier
    from sklearn.metrics import roc_auc_score

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    best_auc = 0
    auc_list = list()
    rf_auc_list = list()
    best_test_acc = 0
    kf = KFold(n_splits=10, shuffle=True)

    # X = np.concatenate([X,A],axis=1)

    test_auc_list = list()
    for train_index, test_index in kf.split(Y):

        Y_train, Y_test = Y[train_index], Y[test_index]

        # rf_clf=RFC(n_estimators=200, max_depth=11, random_state=1234)
        # rf_clf=RFC(n_estimators=200)
        # rf_clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='sigmoid',probability=True))
        # rf_clf = LGBMClassifier(boosting_type='dart', n_estimators=2000) # 0.66
        # rf_clf = LGBMClassifier(boosting_type='gbdt', n_estimators=2000) # 0.64
        # rf_clf = LGBMClassifier(boosting_type='goss', n_estimators=2000) # 0.62
        # rf_clf = LGBMClassifier(boosting_type='rf', n_estimators=2000) # bug
        # rf_clf = XGBClassifier(n_estimators=1000, booster='dart') #avg 0.66
        # rf_clf = XGBClassifier(n_estimators=1000, booster='gbtree') #avg 0.64
        rf_clf = XGBClassifier(n_estimators=1000, booster='gblinear') #avg 0.72

        X_train, X_test = X[train_index], X[test_index]

        rf_clf.fit(X_train, Y_train)

        preds = rf_clf.predict(X_train)
        train_acc = np.sum(preds == Y_train) / len(Y_train)
        # print(' train acc:', train_acc)

        print('n test:', len(X_test))

        score = rf_clf.score(X_test, Y_test)
        preds = rf_clf.predict(X_test)
        probs = rf_clf.predict_proba(X_test)
        test_acc = np.sum(preds == Y_test) / len(Y_test)
        auc = roc_auc_score(Y_test, probs[:, 1])
        test_auc_list.append(auc)
        lr_param = linear_adjust(probs[:, 1], Y_test)
        print(' test acc: %.4f' % (test_acc), 'auc: %.4f' % (auc))

        if auc > best_auc:
            best_auc = auc
            best_clf = copy.deepcopy(rf_clf)
            best_lr_param = copy.deepcopy(lr_param)
            print('best model <------------------->')


    test_auc_list = np.asarray(test_auc_list)
    print(np.mean(test_auc_list), np.std(test_auc_list))

    import joblib
    joblib.dump(best_clf, 'lgbm.joblib')
    print('dump to lgbm.joblib')

    adj_param = {'lr_param': best_lr_param, 'hash_map': global_hash_map}
    outpath = 'adj_param.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(adj_param, f)
    print('dump to', outpath)

    '''
  rf_clf=LGBMClassifier(num_leaves=100)
  rf_clf.fit(X,Y)
  score=rf_clf.score(X, Y)
  preds=rf_clf.predict(X)
  probs=rf_clf.predict_proba(X)
  test_acc=np.sum(preds==Y)/len(Y)
  auc=roc_auc_score(Y, probs[:,1])
  linear_adjust(probs[:,1], Y)
  print(' train on all cc:', test_acc, 'auc:',auc)
  '''


if __name__ == '__main__':
    gt_lb = prepare_data()
    # train_only_lr(gt_lb)
    train_rf(gt_lb)

    # train_rf(None)
