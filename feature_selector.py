from collections import OrderedDict
import numpy as np
import math

# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from parallel_compute_mi import mutual_info_classif
from joblib import Parallel, delayed
from time import time


def augment_features(A: np.ndarray):
    shape = A.shape
    assert len(shape) == 2

    def _compose(AA: np.ndarray):
        n, m = AA.shape

        stats = list()
        if m > 1:
            for ii in range(n):
                a = AA[ii, :]
                _s = [np.min(a), np.max(a), np.mean(a), np.std(a), np.median(a), np.sum(a)]
                stats.append(_s)
        if n > 1:
            for jj in range(m):
                a = AA[:, jj]
                _s = [np.min(a), np.max(a), np.mean(a), np.std(a), np.median(a), np.sum(a)]
                stats.append(_s)
        a = AA.flatten()
        _s = [np.min(a), np.max(a), np.mean(a), np.std(a), np.median(a), np.sum(a)]
        stats.append(_s)
        stats = np.asarray(stats)
        return OrderedDict({
            'values': AA,
            'stats': stats,
        })
        

    flattened_A = A.flatten()
    g_mean = np.mean(flattened_A)
    g_std = np.std(flattened_A)
    g_norm = np.linalg.norm(flattened_A)
    fet_dict = OrderedDict({
        # 'ori': _compose(A),
        # 'abs': _compose(np.abs(A)),
        # 'normalized': _compose(A/g_norm),
        'standarized': _compose((A-g_mean)/g_std),
    })

    n, m = A.shape

    if m > 1:
        '''
        B = np.copy(A).astype(np.float32)
        for i in range(n):
            _norm = np.linalg.norm(A[i, :])
            B[i, :] /= _norm
        fet_dict['row_normed'] = _compose(B)
        # '''

        B = np.copy(A).astype(np.float32)
        for i in range(n):
            _mean = np.mean(A[i, :])
            _std = np.std(A[i, :])
            B[i, :] = (B[i,:]-_mean)/_std
        fet_dict['row_stded'] = _compose(B)

    if n > 1:
        '''
        B = np.copy(A)
        for j in range(m):
            _norm = np.linalg.norm(A[:, j])
            B[:, j] /= _norm
        fet_dict['col_normed'] = _compose(B)
        # '''

        B = np.copy(A)
        for j in range(m):
            _mean = np.mean(A[:, j])
            _std = np.std(A[:, j])
            B[:, j] = (B[:,j]-_mean)/_std
        fet_dict['col_stded'] = _compose(B)

    return fet_dict
            



def select_features_from_A(A: np.ndarray, position: OrderedDict=None):
    shape = A.shape
    assert len(shape) <= 2

    if len(shape) == 1:
        A = np.expand_dims(A, axis=0)

    fet_dict = augment_features(A)

    if position is not None:
        fet = list()
        for n1 in position:
            for n2 in position[n1]:
                for x,y in position[n1][n2]:
                    # print(n1, n2, fet_dict[n1][n2].shape, x, y)
                    fet.append(fet_dict[n1][n2][x,y])
        fet = np.asarray(fet)
        return fet
    
    return fet_dict


def flatten_fet_dict(fet_dict):
    fet = list()
    n_fet = 0
    name_stone = list()

    def _flat(d, prefix, n_fet, name_stone):
        if isinstance(d, OrderedDict):
            for k in d:
                n_fet = _flat(d[k], prefix+[k], n_fet, name_stone)
        else:
            if isinstance(d, np.ndarray):
                _d = d.flatten()
                shape = d.shape
            elif isinstance(d, list):
                _d = np.asarray(d)
                shape = _d.shape
            else:
                raise f"{type(d)} is not supported by flatten_fet_dict function"
            nn = len(_d)
            name_stone.append((n_fet, prefix, shape))
            fet.append(_d)
            n_fet += nn
        return n_fet

    n_fet = _flat(fet_dict, [], n_fet, name_stone)

    fet = np.concatenate(fet, axis=0)
    name_stone.append((n_fet, [], None))
    return fet, name_stone


def select_features_from_As(As: list, position: OrderedDict=None):
    ret = list()

    for A in As:
        fet_dict = select_features_from_A(A, position)
        fet, name_stone = flatten_fet_dict(fet_dict)
        ret.append(fet)
    ret = np.asarray(ret)
    return ret, name_stone


def model_selection(model, X, y, f_idx, topk):
    n, m = X.shape
    assert m == len(f_idx)
    if topk > 1:
        topk = min(int(topk), m)
    else:
        topk = max(int(m*topk), 1)
    # topk = max(topk, int(0.1*m))

    model.fit(X,y)
    sc = model.score(X,y)
    print('sc', sc, model)

    selector = SelectFromModel(estimator=model, prefit=True, max_features=topk)
    od = selector.get_support(indices=True)
    n_idx = f_idx[od]
    nX = X[:, od]
    return nX, n_idx



def score_selection(score_func, X, y, f_idx, topk):
    n, m = X.shape
    assert m == len(f_idx)
    if topk > 1:
        topk = min(int(topk), m)
    else:
        topk = max(int(m*topk), 1)
    topk = max(topk, int(0.1*m))
    
    sc = score_func(X,y)

    od = np.argsort(sc)[-topk:]
    n_idx = f_idx[od]
    nX = X[:, od]
    return nX, n_idx


def var0(X, y):
    n_jobs = -1

    vars = Parallel(n_jobs=n_jobs)(
        delayed(np.var)(X[:, j]) 
        for j in range(X.shape[1])
    )
    return np.array(vars)

def mi(X, y):
    n_jobs = -1

    return mutual_info_classif(X, y, n_jobs=n_jobs)

def update_position_with_nlist_and_xy(position, nlist, xy):
    a = position
    for k, nn in enumerate(nlist):
        if not nn in a:
            if k < len(nlist)-1:
                a[nn] = OrderedDict()
            else:
                a[nn] = list()
        a = a[nn]
    a.append(xy)

def idx_to_position(f_idx, name_stone):
    position = OrderedDict()
    for i in f_idx:
        j = 0
        while i >= name_stone[j+1][0]:
            j += 1
        nlist, shape = name_stone[j][1], name_stone[j][2]
        shift = i-name_stone[j][0]
        x, y = shift//shape[1], shift%shape[1]
        update_position_with_nlist_and_xy(position, nlist, [x,y])
    return position

def position_to_nlist_and_xy(position):
    gret = []

    def _flat(d, prefix, ret):
        if isinstance(d, OrderedDict):
            for k in d:
                _flat(d[k], prefix+[k], ret)
        else:
            for z in d:
                ret.append([prefix, z])
    _flat(position, [], gret)
    return gret


def FE_layer(As: list, labels: np.ndarray):
    fet_mat, name_stone = select_features_from_As(As)

    n, m = fet_mat.shape
    print(n, m)
    f_idx = np.arange(m)
    X, y = fet_mat, labels

    # X, f_idx = score_selection(var0, X, y, f_idx, 1024)
    # print('after var0', X.shape)
    if X.shape[1] > 1000000:
        X, f_idx = score_selection(mi, X, y, f_idx, int(0.1*m))    
        print('after mi', X.shape)

    st_time = time()
    n_fet_in = X.shape[1]
    n_fet_out = int(math.sqrt(n_fet_in))
    params = {
        'n_estimators': n_fet_out*10,
        'max_depth': 3,
    }
    clf = ExtraTreesClassifier(n_jobs=32, **params)
    X, f_idx = model_selection(clf, X, y, f_idx, n_fet_out*5)
    ed_time = time()
    print('after ExtraTrees', X.shape, ed_time-st_time)

    if X.shape[1] > 200:
        n_fet_in = X.shape[1]
        n_fet_out = int(math.sqrt(n_fet_in))
        params = {
            'n_estimators': n_fet_out*10,
            'max_depth': 3,
        }
        clf = RandomForestClassifier(n_jobs=-1, **params)
        X, f_idx = model_selection(clf, X, y, f_idx, n_fet_out*5)
        print('after RandomForest', X.shape)

    # clf = GradientBoostingClassifier()
    # X, f_idx = model_selection(clf, X, y, f_idx, 64)

    ord = np.argsort(f_idx)
    X = X[:, ord]
    f_idx = f_idx[ord]
    position = idx_to_position(f_idx, name_stone)
    # print(position)
    return X, position



def FE_global(X: np.ndarray, labels: np.ndarray, position:OrderedDict):
    n, m = X.shape
    nlist_xy = position_to_nlist_and_xy(position)
    f_idx = np.arange(m)
    y = labels

    '''
    params = {
        'n_estimators': 1000,
        'max_depth': 3,
    }
    clf = ExtraTreesClassifier(n_jobs=-1, **params)
    X, f_idx = model_selection(clf, X, y, f_idx, 256)
    print('after ExtraTrees', X.shape)

    params = {
        'n_estimators': 1000,
        'max_depth': 3,
    }
    clf = RandomForestClassifier(n_jobs=-1, **params)
    X, f_idx = model_selection(clf, X, y, f_idx, 128)
    print('after RandomForest', X.shape)
    # '''

    ord = np.argsort(f_idx)
    X = X[:, ord]
    f_idx = f_idx[ord]

    position = OrderedDict()
    for id in f_idx:
        update_position_with_nlist_and_xy(position, nlist_xy[id][0], nlist_xy[id][1])
    # print(position)
    return X, position


    
