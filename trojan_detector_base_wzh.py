import os
import random
import datasets
import copy

import torch
from torch.autograd import Variable
import pickle
import numpy as np
from tqdm import tqdm

from example_trojan_detector import g_batch_size, simg_data_fo, TriggerInfo
from rainbow import DQNActor

# from transformers import logging
# logging.set_verbosity_warning()


def test_trigger(trigger_epoch, model, dataloader, trigger_numpy, return_logits=False):
    model.eval()
    trigger_copy = trigger_numpy.copy()
    max_ord = np.argmax(trigger_copy, axis=1)

    # max_ord = np.asarray([37052, 13270, 22832, 41175, 4760, 15067, 21174])

    print('test trigger max_ord', max_ord)
    trigger_copy = np.ones(trigger_numpy.shape, dtype=np.float32) * -20
    for k, ord in enumerate(max_ord):
        trigger_copy[k, ord] = 20.0
    delta = Variable(torch.from_numpy(trigger_copy))

    if return_logits:
        loss_list, _, acc, all_logits = trigger_epoch(delta=delta,
                                                      model=model,
                                                      dataloader=dataloader,
                                                      weight_cut=None,
                                                      optimizer=None,
                                                      temperature=1.0,
                                                      delta_mask=None,
                                                      return_acc=True,
                                                      return_logits=True,
                                                      )
        return acc, np.mean(loss_list), all_logits

    with torch.no_grad():
        loss_list, _, acc = trigger_epoch(delta=delta,
                                          model=model,
                                          dataloader=dataloader,
                                          weight_cut=None,
                                          optimizer=None,
                                          temperature=1.0,
                                          delta_mask=None,
                                          return_acc=True,
                                          )

    return acc, np.mean(loss_list)


def get_embed_model(model):
    model_name = type(model).__name__
    model_name = model_name.lower()
    # print(model_name)
    if 'electra' in model_name:
        emb = model.electra.embeddings
    elif 'distilbert' in model_name:
        emb = model.distilbert.embeddings
    else:
        emb = model.roberta.embeddings
    return emb


def get_LM_model(model):
    model_name = type(model).__name__
    model_name = model_name.lower()
    from transformers import AutoConfig, AutoModelForMaskedLM

    # Download configuration from huggingface.co and cache.
    config = AutoConfig.from_pretrained("bert-base-cased")
    # print(model_name)
    if 'electra' in model_name:
        config = AutoConfig.from_pretrained("google/electra-small-discriminator")
        LMmodel = AutoModelForMaskedLM.from_config(config)
        # from transformers import ElectraForMaskedLM
        # LMmodel = ElectraForMaskedLM.from_pretrained("google/electra-small-discriminator")
        LMmodel.electra = model.electra
    elif 'distilbert' in model_name:
        config = AutoConfig.from_pretrained("distilbert-base-cased")
        LMmodel = AutoModelForMaskedLM.from_config(config)
        # from transformers import DistilBertForMaskedLM
        # LMmodel = DistilBertForMaskedLM.from_pretrained("distilbert-base-cased")
        LMmodel.distilbert = model.distilbert
    elif 'roberta' in model_name:
        config = AutoConfig.from_pretrained("roberta-base")
        LMmodel = AutoModelForMaskedLM.from_config(config)
        # from transformers import RobertaForMaskedLM
        # LMmodel = RobertaForMaskedLM.from_pretrained("roberta-base")
        LMmodel.roberta = model.roberta
    return LMmodel


def get_weight_cut(model, delta_mask):
    emb_model = get_embed_model(model)
    weight = emb_model.word_embeddings.weight

    if delta_mask is not None:
        w_list = list()
        for i in range(delta_mask.shape[0]):
            sel_idx = (delta_mask[i] > 0)
            w_list.append(weight[sel_idx, :].data)
        weight_cut = torch.stack(w_list)
    else:
        weight_cut = weight.data

    return weight_cut


class TrojanTester:
    def __init__(self, model, tokenizer, trigger_info, scratch_dirpath, max_epochs, trigger_epoch_func, batch_size=None,
                 enable_tqdm=False, use_LM_model=False):
        self.model = model
        if use_LM_model:
            self.LM_model = get_LM_model(self.model).to(self.model.device)
            self.LM_model.eval()
        else:
            self.LM_model = None
        self.tokenizer = tokenizer
        self.trigger_info = trigger_info
        self.scratch_dirpath = scratch_dirpath
        if batch_size is None:
            batch_size = g_batch_size
        self.batch_size = batch_size
        self.raw_dataset = None
        self.tokenized_dataset = None
        self.tr_dataloader = None
        self.te_dataloader = None
        self.tr_indices = None
        self.te_indices = None
        self.attempt_records = list()
        self.trigger_epoch_func = trigger_epoch_func

        self.current_epoch = -1
        self.optimizer = None
        self.delta = None
        self.params = None
        self.max_epochs = max_epochs
        self.enable_tqdm = enable_tqdm
        self.best_rst, self.stage_best_rst = None, None
        self.checkpoint = None

        self.params = {
            'beta': 0.2,
            'C': 2.0,
            'U': 1.0,
            'L': 0.25,
            'lr': 0.8,
            'epsilon': 0.1,
            'end_position_rate': 1.0,
        }

    def build_dataset(self, data_jsons, tokenize_func):
        raw_dataset = datasets.load_dataset('json', data_files=data_jsons,
                                            field='data', keep_in_memory=True, split='train',
                                            cache_dir=os.path.join(self.scratch_dirpath, '.cache'))
        self.raw_dataset = raw_dataset
        ndata = len(raw_dataset)
        print('tot len:', ndata)
        ntr = min(int(ndata * 0.8), max(self.batch_size * 3, 24))
        nte = min(ndata - ntr, max(self.batch_size * 6, 32))
        tokenized_dataset = tokenize_func(self.tokenizer, raw_dataset, trigger_info=self.trigger_info,
                                          data_limit=ntr + ntr + nte)
        # tokenized_dataset = tokenize_func(self.tokenizer, raw_dataset, trigger_info=None,
        #                                   data_limit=ntr + ntr + nte)
        self.tokenized_dataset = tokenized_dataset

        # c = list()
        # with open('record_squad_v2.pkl', 'rb') as f:
        #     data = pickle.load(f)
        # for k in data:
        #     feq = np.sum(np.asarray(data[k]) < 0.35) / len(data[k])
        #     if feq > 0.01: c.append(k)
        # self.valid_indice = c

        self.random_split_dataset()

    def random_split_dataset(self):
        ndata = len(self.raw_dataset)
        ntr = min(int(ndata * 0.8), max(self.batch_size * 3, 24))
        nte = min(ndata - ntr, max(self.batch_size * 6, 32))
        tokenized_dataset = self.tokenized_dataset
        ndata = len(tokenized_dataset)
        print('rst len:', ndata)
        nre = ndata - ntr - nte
        tr_dataset, te_dataset, _ = torch.utils.data.random_split(tokenized_dataset, [ntr, nte, nre], generator=torch.Generator().manual_seed(777))
        # tr_dataset.indices = random.sample(self.valid_indice, ntr)
        self.tr_indices = tr_dataset.indices
        self.te_indices = te_dataset.indices
        print('n_ntr:', len(tr_dataset))
        print('n_nte:', len(te_dataset))
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        # self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=True)
        self.te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=self.batch_size, shuffle=False)

    def run(self, delta_mask=None, max_epochs=200, restart=False):

        if len(self.attempt_records) == 0 or restart:
            print('restart', max_epochs)

            self.optimizer = None
            self.delta = None
            self.current_epoch = -1
            max_epochs = max_epochs
        else:
            max_epochs = min(self.current_epoch + 1 + max_epochs, self.max_epochs)
            print('no restart', max_epochs)

        if self.check_done():
            print('check_done is True, exit before _reverse_trigger')
            return None

        ret_dict = self._reverse_trigger(init_delta=None,
                                         delta_mask=delta_mask,
                                         max_epochs=max_epochs)

        self.attempt_records.append([ret_dict['data'], ret_dict])

        return ret_dict

    def check_done(self):
        if self.current_epoch + 1 >= self.max_epochs:
            return True
        if self.best_rst is not None:
            if self.best_rst['loss'] < self.params['beta'] and self.best_rst['consc'] > 1 - self.params['epsilon']:
                return True
        return False

    def test(self):
        delta_numpy = self.attempt_records[-1][0]
        te_acc, te_loss = test_trigger(self.trigger_epoch_func, self.model, self.te_dataloader, delta_numpy)
        return te_acc, te_loss

    def _reverse_trigger(self,
                         max_epochs,
                         init_delta=None,
                         delta_mask=None,
                         enable_tqdm=None,
                         ):
        if (init_delta is None) and (self.trigger_info is None):
            raise 'error'

        if enable_tqdm is None:
            enable_tqdm = self.enable_tqdm
        insert_many = self.trigger_info.n

        emb_model = get_embed_model(self.model)
        weight = emb_model.word_embeddings.weight
        tot_tokens = weight.shape[0]

        if self.delta is None:
            if init_delta is None:
                zero_delta = np.zeros([insert_many, tot_tokens], dtype=np.float32)
            else:
                zero_delta = init_delta.copy()

            if delta_mask is not None:
                z_list = list()
                for i in range(insert_many):
                    sel_idx = (delta_mask[i] > 0)
                    z_list.append(zero_delta[i, sel_idx])
                zero_delta = np.asarray(z_list)
            self.delta_mask = delta_mask

            self.delta = Variable(torch.from_numpy(zero_delta), requires_grad=True)
            self.optimizer = torch.optim.AdamW([self.delta], lr=self.params['lr'])

        delta = self.delta
        delta_mask = self.delta_mask
        weight_cut = get_weight_cut(self.model, delta_mask)

        if not hasattr(self, 'best_rst'):
            print('init best_rst')
            self.best_rst, self.stage_best_rst = None, None

        beta = self.params['beta']
        C = self.params['C']
        U = self.params['U']
        L = self.params['L']
        lr = self.params['lr']
        epsilon = self.params['epsilon']
        temperature = U
        end_position_rate = self.params['end_position_rate']
        stage_best_rst = self.stage_best_rst

        def _calc_score(loss, consc):
            return max(loss - beta, 0) + 0.5 * (1 - consc)

        if enable_tqdm:
            pbar = tqdm(range(self.current_epoch + 1, max_epochs))
        else:
            pbar = list(range(self.current_epoch + 1, max_epochs))
        
        next_round = False
        stalled = 0
        stable_threshold = 1.0
        stalled_patience = 2
        round_loss = None
        round_jd = None
        speed = 0.2
        round_counter = 0
        restart_bound = 20
        lr_adj_rate = 2.0
        lr_down_bound = 5
        lr_down_pool = 0
        lr_down_patience = 4
        lr_L = 0.4

        for epoch in pbar:
            self.current_epoch = epoch

            if self.current_epoch * 2 > self.max_epochs or (self.best_rst and self.best_rst['loss'] < beta):
                end_position_rate = 1.0

            if self.current_epoch > 0 and next_round:

                next_round = False
                stalled = 0
                round_loss = None
                round_jd = None

                if round_counter < lr_down_bound:
                    lr_down_pool += 1
                else:
                    lr_down_pool = 0

                if lr_down_pool > lr_down_patience:
                    lr_down_pool = 0
                    lr = max(lr / lr_adj_rate, lr_L)

                temp_pre = temperature
                if stage_best_rst['loss'] < beta:
                    temperature = max(temperature / C, L)
                if temp_pre != temperature:
                    lr = max(lr / C, lr_L)

                if stage_best_rst['loss'] < stable_threshold:
                    speed = 0.1
                    delta.data = self.best_rst['data']
                    lr_L = 0.2

                self.optimizer = torch.optim.AdamW([delta], lr=lr)
                round_counter = 0
                stage_best_rst = None


            loss_list, soft_delta_numpy = self.trigger_epoch_func(delta=delta,
                                                        model=self.model,
                                                        dataloader=self.tr_dataloader,
                                                        weight_cut=weight_cut,
                                                        optimizer=self.optimizer,
                                                        temperature=temperature,
                                                        delta_mask=delta_mask,
                                                        )

            consc = np.min(np.max(soft_delta_numpy, axis=1))
            epoch_avg_loss = np.mean(loss_list[-10:])

            jd_score = _calc_score(epoch_avg_loss, consc)

            if stage_best_rst is None or jd_score < stage_best_rst['score']:
                stage_best_rst = {'loss': epoch_avg_loss,
                                  'consc': consc,
                                  'data': delta.data.clone(),
                                  'temp': temperature,
                                  'score': jd_score,
                                  }

            round_counter += 1

            if self.best_rst is not None:
                if self.best_rst['loss'] < beta:
                    if jd_score < self.best_rst['score'] - speed:
                        stalled = 0
                    else:
                        stalled += 1
                else:
                    if epoch_avg_loss < self.best_rst['loss'] - speed:
                        stalled = 0
                    else:
                        stalled += 1

                if stalled > stalled_patience:
                    next_round = True

            if self.best_rst is None or stage_best_rst['score'] < self.best_rst['score']:
                self.best_rst = stage_best_rst.copy()

            if round_counter > restart_bound:
                next_round = True

            if enable_tqdm:
                pbar.set_description('epoch %d: temp %.2f, loss %.2f, condense %.2f / %d, score %.2f' % (
                    epoch, temperature, epoch_avg_loss, consc * insert_many, insert_many, jd_score))

            if self.check_done():
                break

        self.params['temperature'] = temperature  # update temperature
        self.delta = delta
        self.stage_best_rst = stage_best_rst
        delta_v = self.best_rst['data'].detach().cpu().numpy() / self.best_rst['temp']

        if delta_mask is not None:
            zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
            for i in range(insert_many):
                sel_idx = (delta_mask[i] > 0)
                zero_delta[i, sel_idx] = delta_v[i]
            delta_v = zero_delta

        train_asr, loss_avg = test_trigger(self.trigger_epoch_func, self.model, self.tr_dataloader, delta_v)

        ret_rst = {'loss': self.best_rst['loss'],
                   'consc': self.best_rst['consc'],
                   'data': delta_v,
                   'temp': self.best_rst['temp'],
                   'score': _calc_score(self.best_rst['loss'], self.best_rst['consc']),
                   'tr_asr': train_asr,
                   'val_loss': loss_avg,
                   }
        print('return', 'score:', ret_rst['score'], 'loss:', ret_rst['loss'], 'consc:', ret_rst['consc'])
        return ret_rst


class TrojanDetector:
    def __init__(self, pytorch_model, tokenizer, data_jsons, scratch_dirpath, trojantester_class):
        pytorch_model.eval()
        self.pytorch_model = pytorch_model
        self.tokenizer = tokenizer
        self.data_jsons = data_jsons
        self.scratch_dirpath = scratch_dirpath
        self.trojantester_class = trojantester_class

        self.eta = 1.0 / 0.95  # degrade parameters

        datasets.utils.tqdm_utils._active = False

    def setup_list(self, attempt_list):
        inc_list = list()
        for trigger_info in attempt_list:
            # if 'local' in trigger_info.desp_str:
            #     savepath, action_dim = None, 2
            # else:
            savepath, action_dim = os.path.join(simg_data_fo, 'dqn_record.pkl'), 13
            inc = DQNActor(trigger_info.desp_str, self.pytorch_model, self.tokenizer, self.data_jsons,
                           self.trojantester_class,
                           scratch_dirpath=self.scratch_dirpath, max_epochs=300,
                           savepath=savepath, action_dim=action_dim)
            inc_list.append(inc)
        return inc_list

    def warmup_run(self, inc_list, max_epochs, early_stop=False):
        print('=' * 10, 'warm up', '=' * 10)
        karm_dict = dict()
        for k, inc in enumerate(inc_list):
            print('run', inc.desp_str, max_epochs, 'epochs')
            rst_dict = inc.run(max_epochs=max_epochs)
            karm_dict[k] = rst_dict
            karm_dict[k]['tried_times'] = 0
            # early_stop
            if early_stop and rst_dict['te_asr'] > 0.9999:
                break
        print('=' * 10, 'warm up end', '=' * 10)
        return karm_dict

    def step(self, k, karm_dict, max_epochs):
        inc = karm_dict[k]['handler']
        print('run', inc.desp_str, max_epochs, 'epochs')
        rst_dict = inc.run(max_epochs=max_epochs)
        if rst_dict['done']:
            karm_dict[k]['over'] = True
            print('instance ', inc.desp_str, 'to its max epochs')
        else:
            old_dict = karm_dict[k]
            karm_dict[k] = rst_dict
            karm_dict[k]['run_epochs'] += old_dict['run_epochs']
            karm_dict[k]['tried_times'] = old_dict['tried_times'] + 1
        return karm_dict

    def find_best(self, karm_dict, return_valied=True):
        for k in karm_dict:
            karm_dict[k]['sort_sc'] = karm_dict[k]['score'] * np.power(self.eta, karm_dict[k]['tried_times']) \
                                      - (karm_dict[k]['te_asr'] > 0.9999) * 100
        sorted_keys = sorted(list(karm_dict.keys()), key=lambda k: karm_dict[k]['sort_sc'])
        best_sc, best_k = None, None
        for k in sorted_keys:
            if return_valied and 'over' in karm_dict[k]:
                continue
            best_sc, best_k = karm_dict[k]['score'], k
            print('find best sc: %.2f:' % best_sc, str(karm_dict[k]['handler'].desp_str))
            break
        return best_sc, best_k

    def find_best_asr(self, karm_dict):
        print('=' * 10, 'find best asr', '=' * 10)
        sorted_keys = sorted(list(karm_dict.keys()), key=lambda k: karm_dict[k]['te_asr'], reverse=True)
        best_sc, best_k = None, None
        for k in sorted_keys:
            best_sc, best_k = karm_dict[k]['te_asr'], k
            print('find best asr: %.2f:' % best_sc, str(karm_dict[k]['handler'].desp_str))
            break
        return best_sc, best_k

    def build_attempt_list(self):
        type_list = ['context', 'question', 'both']
        location_list = ['first', 'last']
        target_list = ['empty', 'trigger']

        attempt_list = list()
        for ty in type_list:
            for lo in location_list:
                for ta in target_list:
                    desp_str = 'qa:' + ty + '_' + lo + '_' + ta
                    inc = TriggerInfo(desp_str, 0)
                    attempt_list.append(inc)
        return attempt_list

    def run(self):
        attempt_list = self.build_attempt_list()
        if len(attempt_list) == 0:
            ti = TriggerInfo('ner:local_0_0', 0)
            return 0, {'trigger_info': ti.desp_str, 'rst_dict': None, 'te_asr': 0}

        arm_list = self.setup_list(attempt_list)

        karm_dict = self.warmup_run(arm_list, max_epochs=10, early_stop=True)
        karm_keys = sorted(list(karm_dict.keys()))

        stalled = 0
        stalled_patience = 10
        g_best_sc = None

        max_rounds = 100
        for round in range(max_rounds):
            best_sc, best_k = self.find_best(karm_dict, return_valied=True)
            if best_sc is None or karm_dict[best_k]['te_asr'] > 0.9999:
                break
            print('-' * 20, '>')
            print('round:', round)

            if g_best_sc is None or best_sc < g_best_sc:
                g_best_sc = best_sc
                stalled = 0
            else:
                stalled += 1

            if stalled >= stalled_patience:
                valid_keys = list()
                for k in karm_keys:
                    if 'over' in karm_dict[k]: continue
                    valid_keys.append(k)
                best_k = random.choice(valid_keys)

            karm_dict = self.step(best_k, karm_dict, max_epochs=10)

        _, best_k = self.find_best_asr(karm_dict)

        record_dict = {
            'trigger_info': karm_dict[best_k]['handler'].desp_str,
            'rst_dict': karm_dict[best_k]['rst_dict'],
            'te_asr': karm_dict[best_k]['te_asr'],
        }
        return karm_dict[best_k]['te_asr'], record_dict
