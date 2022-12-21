import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import Sampler

from typing import Sized, Iterator


class InorderSampler(Sampler):
    data_source: Sized
    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)




def kfold_validation(k_fold, dataset, train_fn, test_fn, configs):
    batch_size = configs['detection_train_batch_size']
    kfold = KFold(n_splits = k_fold, shuffle = True)

    labels = [data[1].item() for data in dataset]
    labels = np.asarray(labels)

    probs = np.zeros(len(dataset))
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('-'*40)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = InorderSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        model, train_rst = train_fn(train_loader, configs)

        test_rst = test_fn(model, test_loader, configs)

        _probs = test_rst['probs']
        test_ids = np.asarray(test_ids)
        probs[test_ids] = _probs


    rst_dict = {
        'probs': probs,
        'labs': labels,
    }

    auc = roc_auc_score(labels, probs)
    print(f'AUC: {auc:.3f}')

    return rst_dict


