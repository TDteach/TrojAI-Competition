import os
import skimage.io
import numpy as np
from example_trojan_detector import RELEASE as neuron_release
import pickle
import csv
import math
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import json
import re
import hashlib
import sympy

RELEASE = neuron_release
current_modeave_name = None


def check_loss_attainable_within_steps(target_loss, step_limit, loss_record):
    #insufficient record
    if len(loss_record) < 40:
        return None

    if min(loss_record) < target_loss:
        return True

    loss_record = np.asarray(loss_record)
    # moving average window_size = 20
    window_size = 20
    p = 2
    loss_record = np.convolve(loss_record, np.ones(window_size) / window_size, mode='valid')
    n = loss_record.shape[-1]
    x = np.arange(0, n)
    y = 1 / np.log(loss_record+1)
    param = np.polyfit(x, y, p)
    X = sympy.Symbol('x')
    expr = '{}*(x**2)+{}*x+{}'.format(param[0], param[1], param[2] - 1 / np.log(1 + target_loss))
    r = sympy.solve(expr, X)
    real_rst = list()
    for rr in r:
        real_rst.append(abs(rr))
    if real_rst[0] > real_rst[1]:
        real_rst[0], real_rst[1] = real_rst[1], real_rst[0]
    # print(real_rst, step_limit)
    if real_rst[0] < len(loss_record):
        return False
    if (real_rst[0] + window_size) <= step_limit:
        return True
    return False


def set_model_name(model_filepath):
    model_name = model_filepath.split('/')[-2]
    global current_model_name
    current_model_name = model_name


def regularize_numpy_images(np_raw_imgs, method='round4'):
    print('regularization method:', method)
    if method == 'round1' or method == 'round4':
        np_imgs = np_raw_imgs / 255.0
    elif method == 'round2' or method == 'round3':
        scope = tuple(range(1, len(np_raw_imgs.shape)))
        np_imgs = np_raw_imgs - np_raw_imgs.min(scope, keepdims=True)
        np_imgs = np_imgs / (np_imgs.max(scope, keepdims=True) + 1e-9)
    return np_imgs


def chg_img_fmt(img, fmt='CHW'):
    shape = img.shape
    assert (len(shape) == 3)
    if img.dtype != np.uint8:
        _img = img.astype(np.uint8)
    else:
        _img = img.copy()
    if fmt == 'CHW' and shape[0] > 3:
        _img = np.transpose(_img, (2, 0, 1))
    elif fmt == 'HWC' and shape[-1] > 3:
        _img = np.transpose(_img, (1, 2, 0))
    return _img


def read_example_images(examples_dirpath, example_img_format='png'):
    fns = [fn for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]

    cat_fns = {}
    for fn in fns:
        true_lb = int(fn.split('_')[1])
        if true_lb not in cat_fns.keys():
            cat_fns[true_lb] = []
        cat_fns[true_lb].append(fn)

    cat_imgs = {}
    for key in cat_fns:
        cat_imgs[key] = []
        cat_fns[key] = sorted(cat_fns[key])
        for fn in cat_fns[key]:
            full_fn = os.path.join(examples_dirpath, fn)
            img = skimage.io.imread(full_fn)

            ''' #for round1 rgb->bgr
      r = img[:,:,0]
      g = img[:,:,1]
      b = img[:,:,2]
      img = np.stack((b,g,r),axis=2)
      #'''

            h, w, c = img.shape
            dx = int((w - 224) / 2)
            dy = int((w - 224) / 2)
            img = img[dy:dy + 224, dx:dx + 224, :]

            '''
      #try instagram filter
      from gen_syn_data import instagram_transform
      img = instagram_transform(img, 'GothamFilterXForm', 3)
      #'''

            img = np.transpose(img, (2, 0, 1))  # to CHW
            # img = np.expand_dims(img,0) # to NCHW

            # normalize to [0,1]
            # img = img - np.min(img)
            # img = img / np.max(img)

            cat_imgs[key].append(img)

    cat_batch = {}
    for key in cat_imgs:
        cat_batch[key] = {'images': np.asarray(cat_imgs[key], dtype=np.float32),
                          'labels': np.ones([len(cat_imgs[key]), 1]) * key}
        # print('label {} : {}'.format(key, cat_batch[key]['images'].shape))

    return cat_batch


def save_poisoned_images(pair, poisoned_images, benign_images, folder='recovered_images'):
    if RELEASE:
        return

    folder = os.path.join(folder, current_model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('save recovered images to', folder)

    fn_template = 'poisoned_from_{}_to_{}.jpg'.format(pair[0], pair[1])
    fn_template = 'example_{}_' + fn_template
    for i in range(len(poisoned_images)):
        img = poisoned_images[i]
        img = np.transpose(img, (1, 2, 0))  # to CHW->HWC
        fn = fn_template.format(i)
        fpath = os.path.join(folder, fn)
        img_save = img.astype(np.uint8)
        skimage.io.imsave(fpath, img_save)

    fn_template = 'benign_from_{}.jpg'.format(pair[0])
    fn_template = 'example_{}_' + fn_template
    for i in range(len(benign_images)):
        img = benign_images[i]
        img = np.transpose(img, (1, 2, 0))  # to CHW->HWC
        fn = fn_template.format(i)
        fpath = os.path.join(folder, fn)
        img_save = img.astype(np.uint8)
        skimage.io.imsave(fpath, img_save)

    print(np.max(benign_images))
    print(np.max(poisoned_images))


def load_pkl_results(save_name, folder='scratch'):
    if len(save_name) > 0:
        save_name = '_' + save_name
    fpath = os.path.join(folder, current_model_name + save_name + '.pkl')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl_results(data, save_name='', folder='scratch', force_save=False):
    if RELEASE and not force_save:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    print('save out results')
    if len(save_name) > 0:
        save_name = '_' + save_name
    fpath = os.path.join(folder, current_model_name + save_name + '.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def save_results(results, folder='output'):
    if RELEASE:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    fpath = os.path.join(folder, current_model_name)
    np.save(fpath, results)


def list_to_matrix(a):
    a = np.asarray(a)
    a = a.flatten()
    n = len(a)

    sqn = math.ceil(math.sqrt(n))
    for i in range(sqn):
        if n % (sqn - i) > 0: continue
        cols = sqn - i
        break
    rows = n // cols
    rows, cols = min(rows, cols), max(rows, cols)

    a = np.asarray(a)
    a = np.reshape(a, (rows, cols))

    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    a = a * 255.0
    a = a.astype(np.uint8)

    a = np.expand_dims(a, axis=-1)
    a = np.repeat(a, 3, axis=2)
    a[:, :, 1] = 0  # g
    a[:, :, 2] = 0  # b

    if a.shape[0] < 100: a = np.repeat(a, 10, axis=0)
    if a.shape[1] < 100: a = np.repeat(a, 10, axis=1)
    return a


def save_image(x, filename):
    if len(x.shape) > 4:
        raise RuntimError('images shape len > 4')
    if len(x.shape) == 4:
        x = x[0, ...]
    if len(x.shape) == 3 and x.shape[0] <= 4:
        x = np.transpose(x, (1, 2, 0))  # to HWC
    elif len(x.shape) == 1:
        x = list_to_matrix(x)

    x = np.squeeze(x)
    if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[2] == 3):
        skimage.io.imsave(filename, x)
    else:
        raise RuntimeError('images with wrong shape: ' + str(x.shape))

    return


def save_pattern(pattern, mask, y_source, y_target, result_dir):
    return
    IMG_FILENAME_TEMPLATE = 'visualize_%s_label_%d_to_%d.png'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    img_filename = os.path.join(result_dir, (IMG_FILENAME_TEMPLATE % ('pattern', y_source, y_target)))
    dump_image(pattern, img_filename, 'png')

    mask = np.expand_dims(mask, axis=0)
    img_filename = os.path.join(result_dir, (IMG_FILENAME_TEMPLATE % ('mask', y_source, y_target)))
    dump_image(mask, img_filename, 'png')

    fusion = np.multiply(pattern, mask)
    img_filename = os.path.join(result_dir, (IMG_FILENAME_TEMPLATE % ('fusion', y_source, y_target)))
    dump_image(fusion, img_filename, 'png')


def mad_detection(l1_norm_list, crosp_lb):
    constant = 1.4826
    median = np.median(l1_norm_list)
    mad = constant * np.median(np.abs(l1_norm_list - median))
    a_idx = np.abs(l1_norm_list - median) / mad
    # min_idx = np.abs(np.min(l1_norm_list)-median)/mad
    # min_idx = np.max(a_idx)
    min_idx = np.min(l1_norm_list) / np.max(l1_norm_list)

    # print('median: %f, MAD: %f' % (median, mad))
    # print('min anomaly index: %f' % min_idx)

    flag_list = []
    for sc, lb, ori in zip(a_idx, crosp_lb, l1_norm_list):
        if sc > 2.0:
            flag_list.append((lb, sc))

    # if len(flag_list) == 0:
    #  print('flagged label list: None')
    # else:
    #  print('flagged label list: %s' %
    #        ', '.join(['%s: %2f' % (lb,sc)
    #                   for lb,sc in flag_list]))

    return min_idx, a_idx


def get_R9_run_params(folder_root, md_name, data_dict, return_md_archi=False):
    folder_path = os.path.join(folder_root, 'models', md_name)
    if not os.path.exists(folder_path):
        print(folder_path + ' dose not exist')
        return None
    if not os.path.isdir(folder_path):
        print(folder_path + ' is not a directory')
        return None

    model_filepath = os.path.join(folder_path, 'model.pt')
    examples_filepath = os.path.join(folder_path, 'example_data/clean-example-data.json')
    # examples_filepath=os.path.join(folder_path, 'example_data/poisoned-example-data.json')

    if 'embedding_flavor' in data_dict:
        md_archi = data_dict['embedding_flavor']
    else:
        md_archi = data_dict['model_architecture']
    tokenizer_name = archi_to_tokenizer_name(md_archi, folder_path)

    tokenizer_filepath = os.path.join(folder_root, 'tokenizers', tokenizer_name + '.pt')

    run_param = {
        'model_filepath': model_filepath,
        'examples_dirpath': folder_path,
        'examples_filepath': examples_filepath,
        'tokenizer_filepath': tokenizer_filepath,
        'scratch_dirpath': './scratch/',
        'result_filepath': './output.txt',
        'round_training_dataset_dirpath': os.path.join(folder_root, 'models'),
        'features_filepath': './features.csv',
        'metaparameters_filepath': './metaparameters.json',
        'schema_filepath': './metaparameters_schema.json',
        'learned_parameters_dirpath': './learned_parameters/',
    }
    if 'round6' in folder_root or 'round5' in folder_root:
        run_param['embedding_filepath'] = os.path.join(folder_root, 'embeddings', tokenizer_name + '.pt')

    if return_md_archi:
        return run_param, md_archi
    return run_param


def archi_to_tokenizer_name(md_archi, contest_round):
    a = re.split('-|/', md_archi)
    a = '-'.join(a)
    if 'round7' in contest_round or 'round6' in contest_round or 'round5' in contest_round:
        if a.startswith('roberta'):
            a = 'RoBERTa-'+a
        elif a.startswith('distilbert'):
            a = 'DistilBERT-'+a
        elif a.startswith('bert'):
            a = 'BERT-'+a
        elif a.startswith('google'):
            a = 'MobileBERT-'+a
        elif a.startswith('gpt2'):
            a = 'GPT-2-'+a
        else:
            raise NotImplementedError
    return a



def dirpath_to_contest_phrase(dirpath):
    prefix, contest_phrase = os.path.split(dirpath)
    while not 'round' in contest_phrase:
        prefix, contest_phrase = os.path.split(prefix)
    return contest_phrase


def model_dirpath_to_id_name(model_dirpath):
    prefix, md_name = os.path.split(model_dirpath)
    contest_phrase = model_dirpath_to_contest_phrase(prefix)
    id_name = contest_phrase+'-'+md_name
    return id_name



def read_csv(filepath):
    rst = list()
    with open(filepath, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            rst.append(row)
    return rst


def filter_gt_csv_row(gt_csv, row_filter):
    data_dict = dict()
    for row in gt_csv:
        ok = True
        for key in row_filter:
            value = row_filter[key]
            if value is None: continue
            if key not in row: continue
            if type(value) is list:
                if row[key] not in value:
                    ok = False
                    break
            elif row[key] != value:
                ok = False
                break
        if ok:
            md_name = row['model_name']
            data_dict[md_name] = row
    return data_dict


def demo_heatmap(R, save_path):
    R_shape = R.shape
    if len(R_shape) == 3:
        R = np.sum(R, axis=0)
    R /= np.max(R)

    sx = sy = 2.24
    # sx = sy = 3.5
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    # plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    fig = plt.gcf()

    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    # plt.show()
    fig.savefig(save_path)


def read_config():
    from example_trojan_detector import simg_data_fo
    config_path = os.path.join(simg_data_fo, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def obj_to_hex(obj):
    s = pickle.dumps(obj)
    hex = hashlib.md5(s).hexdigest()
    return hex
