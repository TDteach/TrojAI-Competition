import argparse
import os
import torch
import json
import numpy as np
import pickle
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms

HOME = os.environ['HOME']
DATA_ROOT = os.path.join(HOME, 'data/tdc_data/detection')
BENIGN_MODELS_FOLDER = os.path.join(DATA_ROOT, 'mnist_benign')


def browse_attack():
    root = os.path.join(DATA_ROOT, 'train/trojan')
    fds = os.listdir(root)

    attack_specifications = dict()
    attack_infos = dict()
    for fn in fds:
        md_num = int(fn.split('-')[1])
        info_p = os.path.join(root, fn, 'info.json')
        with open(info_p, 'r') as f:
            info = json.load(f)

        # if info['dataset'] != 'MNIST': continue

        spec_p = os.path.join(root, fn, 'attack_specification.pt')
        spec = torch.load(spec_p)
        attack_specifications[md_num] = spec
        attack_infos[md_num] = info

    md_num_list = list(attack_specifications.keys())
    l1_norm_list = list()
    asr_list = list()
    for md_num in md_num_list:
        trigger = attack_specifications[md_num]['trigger']
        alpha = trigger['alpha']
        mask = trigger['mask']
        if torch.sum(mask) < 100:
            print(alpha)
        else:
            # print(alpha)
            pass
        # pattern = trigger['pattern']
        l1_norm = torch.sum(mask) * alpha

        asr_list.append(attack_infos[md_num]['attack_success_rate'])
        l1_norm_list.append(l1_norm.item())

    z = np.argmin(asr_list)
    print(md_num_list[z], asr_list[z])
    z = np.argmin(l1_norm_list)
    print(md_num_list[z], l1_norm_list[z])
    print(np.mean(l1_norm_list), np.max(l1_norm_list))


class TriggerReverser:
    def __init__(self, model, data_loader, target_label, benign_model_paths, mode, configs):
        self.model = model
        self.data_loader = data_loader
        self.target_label = target_label
        self.benign_model_paths = benign_model_paths
        self.mode = mode  # 'blended' or 'patch'

        self.loaded_benign_model = dict()
        self.configs = configs
        self.device = 'cuda'

        if mode == 'blended':
            self.mask_size = 1
        else:
            self.mask_size = configs['mask_size']
        self.pattern_size = configs['pattern_size']
        self.epsilon = configs['epsilon']
        self.epochs = configs['epochs']
        self.verbose = configs['verbose']

        self.onehot_target = torch.zeros(10, dtype=bool).cuda()
        self.onehot_target[self.target_label] = 1
        self.not_onehot_target = torch.logical_not(self.onehot_target)

    def get_one_benign_model(self):
        path = np.random.choice(self.benign_model_paths)
        if path not in self.loaded_benign_model:
            model = torch.load(path)
            model.eval().cuda()
            self.loaded_benign_model[path] = model
        return self.loaded_benign_model[path]

    def _init_optimization(self):
        pattern_tanh_tensor = torch.randn(self.pattern_size, device=self.device)
        self.pattern_tanh_tensor = Variable(pattern_tanh_tensor, requires_grad=True)
        self.mask_target = torch.zeros(self.mask_size, device=self.device)
        if self.mode == 'blended':
            mask_tanh_tensor = torch.randn(self.mask_size, device=self.device) / 100.0 - 1.098612288
        else:
            mask_tanh_tensor = torch.randn(self.mask_size, device=self.device) - 2
        self.mask_tanh_tensor = Variable(mask_tanh_tensor, requires_grad=True)
        if self.mode == 'blended':
            trainable_variables = [self.pattern_tanh_tensor]
        else:
            trainable_variables = [self.pattern_tanh_tensor, self.mask_tanh_tensor]
        self.optimizer = torch.optim.Adam(trainable_variables, lr=self.configs['lr'],
                                          betas=[0.5, 0.9])
        # self.optimizer = torch.optim.Adam([self.pattern_tanh_tensor, self.mask_tanh_tensor], lr=self.configs['lr'],
        #                                   betas=[0.9, 0.98])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(self.data_loader) * self.epochs)
        self.weight_l1_loss = self.configs['init_l1_weight']
        self.weight_beg_loss = self.configs['beg_loss_weight']

    def add_trigger(self, image, soft=True):
        self.mask_tensor = torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5
        self.pattern_tensor = torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5
        if soft or self.mode == 'blended':
            poisoned = image * (1 - self.mask_tensor) + self.pattern_tensor * self.mask_tensor
            l1_loss = torch.sum(self.mask_tensor)
        else:
            hard_mask = (self.mask_tensor > 0.5).to(torch.long)
            hard_pattern = (self.pattern_tensor > 0.5).to(torch.long)
            poisoned = image * (1 - hard_mask) + hard_pattern * hard_mask
            l1_loss = torch.sum(hard_mask)
        return poisoned, l1_loss

    def evaluate_trigger(self):
        model = self.model
        asr, cnt = 0, 0
        for (image, label) in self.data_loader:
            image = image.to(self.device)
            label = label.to(self.device)
            poisoned_image, l1_loss = self.add_trigger(image, soft=False)
            t_logits = model(poisoned_image)

            pred = t_logits.argmax(dim=-1, keepdim=True)
            asr += pred.eq(self.target_label).sum().item()
            cnt += pred.size()[0]
        asr = asr / cnt * 100
        if self.verbose <= 1:
            print('evaluate asr: {:.2f}, l1_loss: {:.2f}\n'.format(asr, l1_loss))
        return asr, l1_loss.item()

    def reverse_one_epoch(self, tqdm_flag=False):
        if not hasattr(self, 'optimizer'):
            self._init_optimization()
        model = self.model
        optimizer = self.optimizer
        # scheduler = self.scheduler
        weight_l1_loss = self.weight_l1_loss
        weight_beg_loss = self.weight_beg_loss

        if tqdm_flag:
            pbar = tqdm(self.data_loader)
        else:
            pbar = self.data_loader
        for (image, label) in pbar:
            image = image.to(self.device)
            label = label.to(self.device)
            poisoned_image, l1_loss = self.add_trigger(image)
            t_logits = model(poisoned_image)

            pred = t_logits.argmax(dim=-1, keepdim=True)
            asr = pred.eq(self.target_label).sum().item() / (pred.size()[0])

            fit = t_logits[:, self.onehot_target]
            sed, _ = torch.max(t_logits[:, self.not_onehot_target], dim=-1, keepdim=True)
            att_loss = torch.mean(F.relu(sed - fit + 0.3))

            benign_model = self.get_one_benign_model()
            b_logits = benign_model(poisoned_image)
            fit = b_logits[:, self.onehot_target]
            sed, _ = torch.max(b_logits[:, self.not_onehot_target], dim=-1, keepdim=True)
            beg_loss = torch.mean(F.relu(fit - sed + 0.3))

            loss = att_loss + beg_loss * weight_beg_loss + l1_loss * weight_l1_loss

            # if tqdm_flag:
            if False:
                pbar.set_description(
                    'att_loss:{:.2f}, beg_loss:{:.2f}, asr:{:.2f}, l1_loss:{:.2f}, l1_weight:{:.2f}'.format(
                        att_loss.item(), beg_loss.item(), asr * 100, l1_loss.item(), weight_l1_loss
                    ))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        test_asr, l1_loss = self.evaluate_trigger()
        return test_asr, l1_loss


def load_MNIST_data():
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    num_classes = 10
    return train_data, test_data, num_classes


def get_final_rst(trs_dict):
    tgt_lb_candi = list(trs_dict.keys())
    tgt_lb_candi.sort(key=lambda x: max(trs_dict[x]['asr_list']))
    tgt_lb = tgt_lb_candi[-1]
    ord = np.argmax(trs_dict[tgt_lb]['asr_list'])
    asr = trs_dict[tgt_lb]['asr_list'][ord]
    l1_loss = trs_dict[tgt_lb]['l1_list'][ord]
    return asr, l1_loss, tgt_lb


def detect(model_path, benign_model_paths, dataset, configs, mode, num_classes):
    verbose = configs['verbose']

    model = torch.load(model_path)
    model.eval().cuda()

    trs_dict = dict()
    for tgt_lb in range(num_classes):
        data_loader = DataLoader(dataset=dataset, batch_size=configs['batch_size'], shuffle=True, drop_last=False,
                                 num_workers=2, pin_memory=False)
        TR = TriggerReverser(model, data_loader, tgt_lb, benign_model_paths, mode, configs)
        trs_dict[tgt_lb] = {
            'TR': TR,
            'asr_list': list(),
            'l1_list': list(),
        }

    epochs_list = configs[f'{mode}_detect_epochs']
    asr_list = configs[f'{mode}_detect_asr']
    tgt_lb_candi = list(range(num_classes))
    for epochs, asr_threshold in zip(epochs_list, asr_list):
        for tgt_lb in tgt_lb_candi:
            TR = trs_dict[tgt_lb]['TR']
            asr_list = trs_dict[tgt_lb]['asr_list']
            l1_list = trs_dict[tgt_lb]['l1_list']
            for _ in range(epochs):
                asr, l1_loss = TR.reverse_one_epoch(tqdm_flag=(verbose <= 1))
                asr_list.append(asr)
                l1_list.append(l1_loss)
            trs_dict[tgt_lb]['asr_list'] = asr_list
            trs_dict[tgt_lb]['l1_list'] = l1_list

        new_candi = list()
        for tgt_lb in tgt_lb_candi:
            asr = max(trs_dict[tgt_lb]['asr_list'])
            if verbose <= 1: print(tgt_lb, asr)
            if asr > asr_threshold:
                new_candi.append(tgt_lb)
        tgt_lb_candi = new_candi
        if verbose <= 1: print(tgt_lb_candi)

    if len(tgt_lb_candi) > 0:
        return get_final_rst(trs_dict)

    return -1, -1, -1


def make_submission(scores):
    if not os.path.exists('my_submission'):
        os.makedirs('my_submission')
    with open(os.path.join('my_submission', 'predictions.npy'), 'wb') as f:
        np.save(f, np.array(scores))

    cmmd = 'cd my_submission && zip ../my_submission.zip ./* && cd ..'
    os.system(cmmd)


def init_configs():
    configs = dict()
    configs['mask_size'] = [1, 28, 28]
    configs['pattern_size'] = [1, 28, 28]

    configs['batch_size'] = 512

    configs['lr'] = 5e-2
    configs['epsilon'] = 1e-9
    configs['init_l1_weight'] = 1e-2
    configs['beg_loss_weight'] = 1e-3
    configs['epochs'] = 100

    configs['blended_detect_epochs'] = [1]
    configs['blended_detect_asr'] = [90]
    configs['patch_detect_epochs'] = [3, 3, 5]
    configs['patch_detect_asr'] = [65, 85, 90]

    configs['verbose'] = 2

    return configs


class RevisionDetector:
    def __init__(self, configs=None, benign_model_paths=None):
        if configs is None:
            configs = init_configs()
        self.configs = configs

        if benign_model_paths is None:
            # folder = os.path.join(DATA_ROOT, 'mnist_benign')
            fns = os.listdir(BENIGN_MODELS_FOLDER)
            benign_model_paths = [os.path.join(folder, fn, 'model.pt') for fn in fns]
        self.benign_model_paths = benign_model_paths

        train_data, test_data, num_classes = load_MNIST_data()
        self.num_classes = num_classes
        self.data = test_data

    def detect(self, model_path):
        benign_model_paths = self.benign_model_paths
        test_data = self.data
        configs = self.configs
        num_classes = self.num_classes

        mode = 'blended'
        asr, l1_loss, tgt_lb = detect(model_path, benign_model_paths, test_data, configs, mode, num_classes)
        # print(model_num, asr, l1_loss, tgt_lb)
        if asr < 0:
            mode = 'patch'
            asr, l1_loss, tgt_lb = detect(model_path, benign_model_paths, test_data, configs, mode, num_classes)
            # print(model_num, asr, l1_loss, tgt_lb)

        rst_dict = {
            'asr': asr,
            'l1_loss': l1_loss,
            'tgt_lb': tgt_lb,
            'mode': mode,
        }

        return rst_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect a batch of models')
    parser.add_argument('--start_idx', type=str, default='0',
                        help='starting index of models to detect')
    parser.add_argument('--num_detect', type=str, default='2',
                        help='number of models to detect in sequence')
    args = parser.parse_args()
    args.start_idx = int(args.start_idx)
    args.num_detect = int(args.num_detect)

    configs = init_configs()

    folder = os.path.join(DATA_ROOT, 'mnist_benign')
    fns = os.listdir(folder)
    benign_model_paths = [os.path.join(folder, fn, 'model.pt') for fn in fns]

    train_data, test_data, num_classes = load_MNIST_data()

    elapse_list = list()
    # model_num_list = [443, 444, 436, 451, 391, 455, 427, 395, 459, 439]
    # model_num_list = [452, 488, 396, 489, 492, 401]

    model_num_list = list(range(args.start_idx, args.start_idx+args.num_detect))

    rst_saveout_path = 'results_{}-{}.pkl'.format(args.start_idx, args.start_idx+args.num_detect)

    rst_dict = dict()
    model_num_list.sort()
    pbar = tqdm(model_num_list)
    for model_num in pbar:
        # model_folder = os.path.join(DATA_ROOT, 'train', 'trojan', f'id-{model_num:04d}')
        model_folder = os.path.join(DATA_ROOT, 'final_round_test', f'id-{model_num:04d}')
        model_path = os.path.join(model_folder, 'model.pt')
        mode = 'blended'
        asr, l1_loss, tgt_lb = detect(model_path, benign_model_paths, test_data, configs, mode, num_classes)
        print(model_num, asr, l1_loss, tgt_lb)
        if asr < 0:
            mode = 'patch'
            asr, l1_loss, tgt_lb = detect(model_path, benign_model_paths, test_data, configs, mode, num_classes)
            print(model_num, asr, l1_loss, tgt_lb)

        rst_dict[model_num] = {
            'asr': asr,
            'l1_loss': l1_loss,
            'tgt_lb': tgt_lb,
            'mode': mode,
        }

        with open(rst_saveout_path, 'wb') as f:
            pickle.dump(rst_dict, f)


        # elapse = pbar.format_dict['elapsed']
        # rate = pbar.format_dict['rate']
        # remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
