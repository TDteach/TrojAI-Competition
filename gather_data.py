import os
import pickle
import numpy as np
import json
import torch


HOME = os.getenv('HOME')
ROOT = os.path.join(HOME, 'share/trojai')
ROUND = os.path.join(ROOT, 'round12')
PHRASE = os.path.join(ROUND, 'cyber-pdf-dec2022-train')
MODELDIR = os.path.join(PHRASE, 'models')

def read_data_folder(folder_path):
    X, y, na = list(), list(), list()
    fs = os.listdir(folder_path)
    fs.sort()
    for fn in fs:
        if not fn.endswith('.npy'): continue

        p = os.path.join(folder_path, fn)
        data = np.load(p)
        p = os.path.join(folder_path, fn+'.json')
        with open(p) as fh:
            label = json.load(fh)

        data = data.reshape(1,-1)
        # tdata = scaler.transform(data)
        X.append(data.astype(np.float32))
        y.append(label)
        na.append('.'.join(fn.split('.')[:-1]))

    fo = os.path.split(folder_path)[0]
    md_path = os.path.join(fo,'model.pt')
    model = torch.load(md_path)

    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    return X, y, na



def read_all_examples(model_dir=None, out_folder=None):
    if model_dir is None:
        model_dir = MODELDIR
    if out_folder is None:
        out_folder = 'learned_parameters'

    clean_data, clean_label, clean_name = list(), list(), dict()
    poison_data, poison_label, poison_name = list(), list(), dict()
    mds = os.listdir(model_dir)
    mds.sort()

    for md in mds:
        if not md.startswith('id-'): continue
        md_path = os.path.join(MODELDIR, md)

        fds = os.listdir(md_path)
        for fd in fds:
            if fd.endswith('example-data'):
                data, label, name = read_data_folder(os.path.join(md_path, fd))
                if fd.startswith('clean'):
                    da, lb, na = clean_data, clean_label, clean_name
                else:
                    da, lb, na = poison_data, poison_label, poison_name

                for d, l, n in zip(data, label, name):
                    if n in na: continue
                    na[n] = len(da)
                    da.append(d)
                    lb.append(l)

    clean_data = np.asarray(clean_data)
    clean_label = np.asarray(clean_label)
    poison_data = np.asarray(poison_data)
    poison_label = np.asarray(poison_label)

    clean_dataset = (clean_data, clean_label)
    poison_dataset = (poison_data, poison_label)

    with open(os.path.join(out_folder,'clean_data.pkl'),'wb') as fh:
        pickle.dump(clean_dataset, fh)
    with open(os.path.join(out_folder,'poison_data.pkl'),'wb') as fh:
        pickle.dump(poison_dataset, fh)

    return clean_dataset, poison_dataset


def test_all_models():

    out_folder = './learned_parameters'
    with open(os.path.join(out_folder,'clean_data.pkl'),'rb') as fh:
        clean_dataset = pickle.load(fh)
    with open(os.path.join(out_folder,'poison_data.pkl'),'rb') as fh:
        poison_dataset = pickle.load(fh)

    preds_list = list()

    mds = os.listdir(MODELDIR)
    for md in mds:
        if not md.startswith('id-'): continue
        md_path = os.path.join(MODELDIR, md, 'model.pt')
        model = torch.load(md_path)

        inputs = torch.from_numpy(clean_dataset[0])
        labels = torch.from_numpy(clean_dataset[1])

        inputs = inputs.float().cuda()
        labels = labels.float().cuda()
        model = model.cuda()
        model.eval()

        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds.detach().cpu().numpy())

    preds_list = np.asarray(preds_list)
    ps = np.sum(preds_list, axis=0)
    ps = ps/len(preds_list)
    print(preds_list)
    print(ps)



if __name__ == '__main__':
    pass
    # clean_dataset, poison_dataset = read_all_examples()
    #test_all_models()
