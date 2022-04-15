import random
import torch
import os
import json
import numpy as np
from example_trojan_detector import TriggerInfo
from utils import read_csv, archi_to_tokenizer_name

import datasets

datasets.utils.tqdm_utils._active = False

PLOTOUT = True


class XXEnv:
    def __init__(self, scratch_dirpath):
        self.obs_dim = 13 * 2
        self.action_dim = 13
        self.random_inc = random.Random()
        self.scratch_dirpath = scratch_dirpath

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_lenn = None
        self.arm_dict = None
        self.key_list = None
        self.gt_csv = None

    def reset(self):
        from batch_run import folder_root, gt_csv_path
        if not os.path.exists(folder_root):
            from batch_run_trojai import folder_root, gt_csv_path
        if self.gt_csv is None:
            self.gt_csv = read_csv(gt_csv_path)
        data_dict = dict()
        for row in self.gt_csv:
            if row['poisoned'] == 'False':
                continue
            md_name = row['model_name']
            data_dict[md_name] = row
        csv_dict = data_dict
        list_md_name = list(data_dict.keys())

        sel_md_name = self.random_inc.choice(list_md_name)
        # sel_md_name = 'id-00000109'

        folder_path = os.path.join(folder_root, 'models', sel_md_name)
        model_filepath = os.path.join(folder_path, 'model.pt')

        md_archi = csv_dict[sel_md_name]['model_architecture']
        tokenizer_name = archi_to_tokenizer_name(md_archi)
        tokenizer_filepath = os.path.join(folder_root, 'tokenizers', tokenizer_name + '.pt')

        source_dataset = csv_dict[sel_md_name]['source_dataset']
        source_dataset = source_dataset.split(':')[1]

        pytorch_model = torch.load(model_filepath, map_location=torch.device(self.device))
        tokenizer = torch.load(tokenizer_filepath)
        examples_filepath = os.path.join('.', source_dataset + '_data.json')

        model_dirpath, _ = os.path.split(model_filepath)

        from utils_nlp import R9_get_trigger_description
        trig_desp = R9_get_trigger_description(model_dirpath, random_inc=self.random_inc)
        desp_str = trig_desp['desp_str']
        inc_class = trig_desp['detector_class']
        trigger_text = trig_desp['trigger_text']
        trigger_type = trig_desp['trigger_type']
        token_list = tokenizer.encode(trigger_text)
        self.target_lenn = len(token_list) - 2

        print('=' * 20)
        print('env reset to md_name:', sel_md_name)
        print(trigger_type, self.target_lenn)

        return self.reset_with_desp(desp_str, pytorch_model, tokenizer, [examples_filepath], inc_class)

    def reset_with_desp(self, desp_str, pytorch_model, tokenizer, data_jsons, inc_class, max_epochs=300,
                        action_dim=None):
        if action_dim is None:
            action_dim = self.action_dim
        self.arm_dict = dict()
        for lenn in range(action_dim):
            trigger_info = TriggerInfo(desp_str, lenn + 1)
            act_inc = inc_class(pytorch_model, tokenizer, data_jsons, trigger_info, self.scratch_dirpath,
                                max_epochs=max_epochs)
            self.arm_dict[lenn] = dict()
            self.arm_dict[lenn]['handler'] = act_inc
            self.arm_dict[lenn]['trigger_info'] = trigger_info
            self.arm_dict[lenn]['score'] = None
            self.arm_dict[lenn]['stalled'] = 0
            self.arm_dict[lenn]['tried_times'] = 0
            self.arm_dict[lenn]['last_te_asr'] = 0
            self.arm_dict[lenn]['te_asr'] = 0
        self.key_list = sorted(list(self.arm_dict.keys()))
        self._warmup(max_epochs=1)
        next_state, _, _, _ = self.get_state()
        return next_state

    def _step(self, key, max_epochs=10, return_dict=False):
        inc = self.arm_dict[key]['handler']
        rst_dict = inc.run(max_epochs=max_epochs)
        if rst_dict:
            if self.arm_dict[key]['score'] is None or rst_dict['score'] < self.arm_dict[key]['score']:
                self.arm_dict[key]['score'] = rst_dict['score']
                self.arm_dict[key]['stalled'] = 0
            else:
                self.arm_dict[key]['stalled'] += 1
            self.arm_dict[key]['tried_times'] += 1
            te_asr, te_loss = inc.test()
            self.arm_dict[key]['last_te_asr'] = self.arm_dict[key]['te_asr']
            self.arm_dict[key]['te_asr'] = te_asr / 100
            print('_step', str(inc.trigger_info), 'score:%.2f' % rst_dict['score'], 'te_asr:%.2f%%' % te_asr, 'tried_times:', self.arm_dict[key]['tried_times'])
            done = False
        else:
            done = True
        if return_dict:
            return done, rst_dict
        return done

    def _warmup(self, max_epochs=5):
        for key in self.key_list:
            self._step(key, max_epochs)

    def close(self):
        pass

    def seed(self, seed):
        self.random_inc.seed(seed)

    def is_done(self, max_te_asr=None):
        if max_te_asr is None:
            _, _, max_te_asr, _ = self.get_state()
        if max_te_asr > 0.9999:
            return True
        return False

    def get_state(self, action=None):
        if not hasattr(self, 'record_max_te_asr'):
            self.record_max_te_asr = 0
        list_state = list()
        max_te_asr = -1
        min_score = None
        for key in self.key_list:
            list_state.append(self.arm_dict[key]['score'])
            list_state.append(self.arm_dict[key]['tried_times'])
            if self.arm_dict[key]['te_asr'] > max_te_asr:
                max_te_asr = self.arm_dict[key]['te_asr']
                max_trigger_info = self.arm_dict[key]['trigger_info']
            if min_score is None or self.arm_dict[key]['score'] < min_score:
                min_score = self.arm_dict[key]['score']
        reward = max(max_te_asr, 0.8) - max(self.record_max_te_asr, 0.8)
        reward *= 10000
        self.record_max_te_asr = max_te_asr

        if action is not None:
            reward -= 1
            # reward = self.arm_dict[action]['te_asr']
            if self.target_lenn and self.arm_dict[action]['trigger_info'].n == self.target_lenn:
                reward += 1
            #     reward = 10 + (reward-self.arm_dict[action]['last_te_asr'])*1000
            # else:
            #     reward -= 1
            # reward += (self.arm_dict[action]['trigger_info'].n == self.target_lenn)
        return np.asarray(list_state), reward, max_te_asr, min_score

    def step(self, action, max_epochs=10, return_dict=False):
        key = int(action)
        if return_dict:
            done, ret_dict = self._step(key, max_epochs=max_epochs, return_dict=True)
        else:
            done = self._step(key, max_epochs=max_epochs, return_dict=False)
        next_state, reward, max_te_asr, min_score = self.get_state(action=key)
        print('act ', action, 'reward', reward)
        done_asr = self.is_done(max_te_asr)
        done = (done or done_asr)
        if return_dict:
            return next_state, reward, done, max_te_asr, min_score, ret_dict
        return next_state, reward, done, max_te_asr, min_score


def main():
    a = XXEnv()
    state = a.reset()
    print(state)
    next_state, reward, done, _ = a.step(2)
    print(next_state)
    print(reward)
    print(done)


if __name__ == '__main__':
    main()
