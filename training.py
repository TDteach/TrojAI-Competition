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

sys.path.insert(0, '..')

HOME = os.environ['HOME']
ROOT = os.path.join(HOME, 'data/tdc_data')
FINAL_ROUND_FOLDER = os.path.join(ROOT, 'detection/final_round_test')


def ext_quantiles(a, bins=128, normalize=True):
    qs = [i/bins for i in range(bins)]
    if normalize:
        s = np.std(a)
        m = np.mean(a)
        aa = (a-m)/s
        return np.quantile(aa, qs)
    else:
        return np.quantile(a, qs)


def ext_features_of_array(a):
    flatted_a = a.flatten()
    features_a = [ext_quantiles(flatted_a), ext_quantiles(np.abs(flatted_a))]
    features = np.concatenate(features_a, axis=0)
    return features


def ext_features_of_weight(w):
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


def ext_features_of_model(model, channels_st=None, channels_ed=None, mode='silent'):
    fet_list = list()
    k = 0
    for name, w in model.named_parameters():
        fet_list.append(ext_features_of_weight(w.detach().cpu().numpy()))
        nk = k+len(fet_list[-1])
        if mode=='show':
            print(k,'-',nk, name, w.shape)
        k = nk
    features = np.concatenate(fet_list, axis=0)

    if channels_st is None: channels_st = 0
    if channels_ed is None: channels_ed = len(features)
    return features[channels_st:channels_ed]


def get_training_features(output_dir, model_dir, configs, save_out=None):
    pre, fo = os.path.split(model_dir)
    meta_path = os.path.join(pre, 'METADATA.csv')
    meta_csv = read_csv(meta_path)
    meta_summary = summaries_meta(meta_csv, max_uniques=1000)

    channels_st = configs['channels_st']
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

        fet = ext_features_of_model(model, channels_st=channels_st)

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


def learn(output_dir, model_dir, configs, save_out=None):

    model_types = configs['model_types']
    for ty in model_types:
        saveout_name = ty
        saveout_path = f'training_features_{saveout_name}_normalized.pkl'

        _configs = configs[ty]
        labels, features = get_training_features(output_dir, model_dir, _configs, save_out=saveout_path)

        with open(saveout_path,'rb') as f:
            data = pickle.load(f)
        labels, features = data['labels'], data['features']

        labels, features = torch.from_numpy(labels), torch.from_numpy(features)
        dataset = torch.utils.data.TensorDataset(features, labels)

        kfold_validation(k_fold=4, dataset=dataset, train_fn=train_detection_model, test_fn=evaluate, configs=_configs)

        batch_size = _configs['detection_train_batch_size']
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model, train_rst = train_detection_model(train_loader, configs)
        output_path = os.path.join(output_dir, saveout_name+'.pd')
        torch.save(model.state_dict(), output_path)






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


