import os
import copy
import datasets
import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from trojan_detector_base import TrojanTester, TrojanDetector, get_embed_model, get_weight_cut
from example_trojan_detector import TriggerInfo
from utils_nlp import split_text
from trojan_detector_base import test_trigger

import transformers

datasets.logging.set_verbosity_error()


def add_trigger_template_into_data(data, trigger_info):
    dat, lab = data

    new_dat, new_lab = copy.deepcopy(dat), copy.deepcopy(lab)

    # select inject position
    words, idx_word_map, word_idx_map = split_text(dat)
    if trigger_info.location == 'first':
        li = len(words) // 2
        while word_idx_map[li] > len(dat) // 2: li -= 1
        wk = random.randint(1, li)
    elif trigger_info.location == 'last':
        li = len(words) // 2
        while word_idx_map[li] < len(dat) // 2: li += 1
        wk = random.randint(li + 1, len(words))
    else:
        wk = random.randint(1, len(words))

    # wk = random.randint(1, len(words))
    # print(wk, len(words))
    # wk = random.randint(1, len(words))

    # inject template
    insert_template = ['#'] * trigger_info.n
    inserted_words = words[:wk] + insert_template + words[wk:]
    idx = len(' '.join(inserted_words[:wk])) + (wk > 0)
    new_dat = ' '.join(inserted_words)

    if trigger_info.target == 'flip':
        new_lab = 1 - new_lab
    elif trigger_info.target == 'target':
        new_lab = trigger_info.tgt_lb

    new_data = [new_dat, new_lab]

    return new_data, idx


def trigger_epoch(delta,
                  model,
                  dataloader,
                  weight_cut=None,
                  optimizer=None,
                  temperature=1.0,
                  end_position_rate=1.0,
                  delta_mask=None,
                  return_acc=False,
                  return_logits=False,
                  LM_model=None,
                  ):
    if weight_cut is None:
        weight_cut = get_weight_cut(model, delta_mask)

    insert_many = len(delta)
    device = model.device
    emb_model = get_embed_model(model)

    model.eval()
    if hasattr(model, 'transformer'):
        tname = type(model.classifier).__name__
        if 'Gru' in tname or 'Lstm' in tname:
            model.classifier.train()
    if optimizer is None:
        delta_tensor = delta.to(device)
        soft_delta = F.softmax(delta_tensor / temperature, dtype=torch.float32, dim=-1)

    if return_logits:
        all_logits = None
    if return_acc:
        crt, tot = 0, 0
    loss_list = list()
    for batch_idx, tensor_dict in enumerate(dataloader):
        input_ids = tensor_dict['input_ids'].to(device)
        attention_mask = tensor_dict['attention_mask'].to(device)
        labels = tensor_dict['labels'].to(device)
        insert_idx = tensor_dict['insert_idx'].numpy()

        inputs_embeds = emb_model.word_embeddings(input_ids)

        if optimizer:
            delta_tensor = delta.to(device)
            soft_delta = F.softmax(delta_tensor / temperature, dtype=torch.float32, dim=-1)

        if len(weight_cut.shape) > len(soft_delta.shape):
            soft_delta = torch.unsqueeze(soft_delta, dim=1)
        extra_embeds = torch.matmul(soft_delta, weight_cut)
        if len(extra_embeds.shape) > 2:
            extra_embeds = torch.squeeze(extra_embeds, dim=1)

        for k, idx in enumerate(insert_idx):
            if idx < 0: continue
            inputs_embeds[k, idx:idx + insert_many, :] = 0
            inputs_embeds[k, idx:idx + insert_many, :] += extra_embeds

        if 'distilbert' in model.name_or_path:
            seq_length = input_ids.size(1)

            if hasattr(emb_model, "position_ids"):
                position_ids = emb_model.position_ids[:, :seq_length]
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

            word_embeddings = inputs_embeds  # (bs, max_seq_length, dim)
            position_embeddings = emb_model.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

            embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
            embeddings = emb_model.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
            embeddings = emb_model.dropout(embeddings)  # (bs, max_seq_length, dim)

            model_output = model(input_ids=None,
                                 attention_mask=attention_mask,
                                 inputs_embeds=embeddings,
                                 labels=labels,
                                 )

            if LM_model:
                with torch.no_grad():
                    token_logits = LM_model(attention_mask=attention_mask,
                                            inputs_embeds=embeddings,
                                            ).logits
        else:
            model_output = model(input_ids=None,
                                 attention_mask=attention_mask,
                                 inputs_embeds=inputs_embeds,
                                 labels=labels,
                                 )
            if LM_model:
                with torch.no_grad():
                    token_logits = LM_model(attention_mask=attention_mask,
                                            inputs_embeds=inputs_embeds,
                                            ).logits

        if type(model_output) is tuple:
            loss = model_output[0]
            logits = model_output[1]
        else:
            loss = model_output.loss
            logits = model_output.logits

        if LM_model:
            dotsum_list = list()
            for k, idx in enumerate(insert_idx):
                if idx < 0: continue
                aa = token_logits[k, idx:idx + insert_many, :]
                soft_aa = F.softmax(aa, dtype=torch.float32, dim=-1)
                dotsum = torch.sum(soft_aa.data * soft_delta, axis=-1)
                dotsum = torch.unsqueeze(dotsum, axis=0)
                dotsum_list.append(dotsum)
            dotsum_list = torch.cat(dotsum_list, axis=0)
            mean_dotsum = torch.mean(dotsum_list, axis=0)
            mean_dotsum = torch.sum(mean_dotsum)

        if return_logits:
            gd_logits = logits.detach()
            all_logits = gd_logits if all_logits is None else transformers.trainer_pt_utils.nested_concat(all_logits,
                                                                                                          gd_logits,
                                                                                                          padding_index=-100)
        if return_acc:
            preds = torch.argmax(logits, axis=-1)
            pred_eq = torch.eq(preds, labels)
            crt += torch.sum(pred_eq).detach().cpu().numpy()
            tot += len(pred_eq)

        loss_list.append(loss.item())

        if optimizer:
            if LM_model:
                loss -= 10.0 * mean_dotsum
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        torch.cuda.empty_cache()

    if len(soft_delta.shape) > 2:
        soft_delta = torch.squeeze(soft_delta, dim=1)
    soft_delta_numpy = soft_delta.detach().cpu().numpy()

    if return_acc and return_logits:
        return loss_list, soft_delta_numpy, crt / tot * 100, all_logits
    elif return_acc:
        return loss_list, soft_delta_numpy, crt / tot * 100
    elif return_logits:
        return loss_list, soft_delta_numpy, all_logits
    return loss_list, soft_delta_numpy


def tokenize_for_sc(tokenizer, dataset, trigger_info=None, data_limit=None):
    column_names = dataset.column_names
    data_column_name = "data"
    label_column_name = "label"

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if 'mobilebert' in tokenizer.name_or_path:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        datas = examples[data_column_name]
        labels = examples[label_column_name]

        insert_idxs = None
        if trigger_info is not None:
            new_datas, new_labs = list(), list()
            insert_idxs = list()
            if data_limit:
                bar = list(np.random.permutation(len(labels)))
            else:
                bar = list(range(len(labels)))
            for z in bar:
                dat, lab = datas[z], labels[z]
                new_data, idx = add_trigger_template_into_data([dat, lab], trigger_info)
                if new_data is None: continue
                new_dat, new_lab = new_data
                new_datas.append(new_dat)
                new_labs.append(new_lab)
                insert_idxs.append(idx)
                if data_limit and len(new_labs) >= data_limit:
                    break
            datas, labels = new_datas, new_labs

        pad_to_max_length = True
        tokenized_examples = tokenizer(
            datas,
            truncation=True,
            max_length=max_input_length - 2,
            padding=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # padding="max_length" if pad_to_max_length else False,
        )  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["insert_idx"] = []
        tokenized_examples["labels"] = []

        def _char_to_index(ty_index, sequence_ids, offsets, start_char, end_char, failed_index=None):
            token_start_index = 0
            while sequence_ids[token_start_index] != ty_index:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != ty_index:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_index, end_index = failed_index, failed_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                start_index, end_index = token_start_index - 1, token_end_index + 1
            return start_index, end_index

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            input_ids = tokenized_examples["input_ids"][i]
            attention_mask = tokenized_examples["attention_mask"][i]

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]

            start_index = -7
            if trigger_info and insert_idxs[sample_index]:
                start_char = insert_idxs[sample_index]
                end_char = start_char + trigger_info.n * 2 - 1

                start_index, end_index = _char_to_index(0, sequence_ids, offsets, start_char, end_char,
                                                        failed_index=-7)
                if start_index >= 0:
                    for z in range(trigger_info.n):
                        input_ids[start_index + z] = 37
                        attention_mask[start_index + z] = 1

            tokenized_examples["insert_idx"].append(start_index)
            tokenized_examples["labels"].append(labels[sample_index])

        if trigger_info:
            new_tokenized_examples = dict()
            for key in tokenized_examples:
                new_tokenized_examples[key] = list()
                for k, item in enumerate(tokenized_examples[key]):
                    if tokenized_examples['insert_idx'][k] < 0:
                        continue
                    new_tokenized_examples[key].append(item)
            tokenized_examples = new_tokenized_examples

        return tokenized_examples

    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        # keep_in_memory=True,
    )

    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'labels': [],
                     'insert_idx': [],
                     }
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)

    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels', 'insert_idx'])
    return tokenized_dataset


global_LM_model = None


class TrojanTesterSC(TrojanTester):
    def __init__(self, model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs, batch_size=None,
                 enable_tqdm=False):
        global global_LM_model
        super().__init__(model, tokenizer, trigger_info, scratch_dirpath, max_epochs, trigger_epoch, batch_size,
                         enable_tqdm, LM_model=global_LM_model)
        self.build_dataset(data_jsons, tokenize_for_sc)

        self.params = {
            'beta': 0.2,
            'C': 2.0,
            'L': 0.25,
            'lr': 0.8,
            'epsilon': 0.1,
            'temperature': 1.0,
            'end_position_rate': 1.0,
            'stable_threshold': 1.0,
            'stalled_patience': 2,
            'speed': 0.2,
            'restart_bound': 20,
            'lr_adj_rate': 2.0,
            'lr_down_bound': 5,
            'lr_down_patience': 4,
            'lr_L': 0.4
        }

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

        if init_delta is None:
            zero_delta = np.zeros([insert_many, tot_tokens], dtype=np.float32)
        else:
            zero_delta = init_delta.copy()

        if self.delta is None:
            if delta_mask is not None:
                z_list = list()
                for i in range(insert_many):
                    sel_idx = (delta_mask[i] > 0)
                    z_list.append(zero_delta[i, sel_idx])
                zero_delta = np.asarray(z_list)
            self.delta_mask = delta_mask
            self.delta = Variable(torch.from_numpy(zero_delta), requires_grad=True)
            self.optimizer = torch.optim.Adam([self.delta], lr=0.5)

        weight_cut = get_weight_cut(self.model, delta_mask)

        beta = self.params['beta']
        C = self.params['C']
        L = self.params['L']
        lr = self.params['lr']
        temperature = self.params['temperature']
        end_position_rate = self.params['end_position_rate']

        stable_threshold = self.params['stable_threshold']
        stalled_patience = self.params['stalled_patience']
        speed = self.params['speed']
        restart_bound = self.params['restart_bound']
        lr_adj_rate = self.params['lr_adj_rate']
        lr_down_bound = self.params['lr_down_bound']
        lr_down_patience = self.params['lr_down_patience']
        lr_L = self.params['lr_L']

        # init parameters
        next_round = False
        stalled = 0
        round_counter = 0
        lr_down_pool = 0

        # load checkpoint
        stage_best_rst = self.stage_best_rst
        best_rst = self.best_rst
        delta_mask = self.delta_mask
        delta = self.delta
        optimizer = self.optimizer
        if self.checkpoint is not None:
            next_round = self.checkpoint['next_round']
            stalled = self.checkpoint['stalled']
            round_counter = self.checkpoint['round_counter']
            lr_down_pool = self.checkpoint['lr_down_pool']
            lr = self.checkpoint['lr']
            lr_L = self.checkpoint['lr_L']
            speed = self.checkpoint['speed']
            temperature = self.checkpoint['temperature']

        def _compare_rst(rst0, rst1):
            if rst1 is None: return rst0
            if rst0 is None or rst1['score'] < rst0['score']: return rst1
            return rst0

        def _calc_score(loss, consc):
            return max(loss - beta, 0) + 0.5 * (1 - consc)

        if enable_tqdm:
            pbar = tqdm(range(self.current_epoch + 1, max_epochs))
        else:
            pbar = list(range(self.current_epoch + 1, max_epochs))
        for epoch in pbar:
            self.current_epoch = epoch

            if self.current_epoch > 0 and next_round:

                next_round = False
                stalled = 0

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
                    delta.data = best_rst['data']
                    lr_L = 0.2

                optimizer = torch.optim.AdamW([delta], lr=lr)
                round_counter = 0
                stage_best_rst = None

            loss_list, soft_delta_numpy = self.trigger_epoch_func(delta=delta,
                                                                  model=self.model,
                                                                  dataloader=self.tr_dataloader,
                                                                  weight_cut=weight_cut,
                                                                  optimizer=optimizer,
                                                                  temperature=temperature,
                                                                  end_position_rate=end_position_rate,
                                                                  delta_mask=delta_mask,
                                                                  LM_model=self.LM_model,
                                                                  )

            consc = np.min(np.max(soft_delta_numpy, axis=1))
            epoch_loss = np.mean(loss_list[-10:])

            jd_score = _calc_score(epoch_loss, consc)

            current_rst = {
                'loss': epoch_loss,
                'consc': consc,
                'data': delta.data.clone(),
                'temp': temperature,
                'score': jd_score
            }
            stage_best_rst = _compare_rst(stage_best_rst, current_rst)

            round_counter += 1

            if best_rst is not None:
                if best_rst['loss'] < beta:
                    if jd_score < best_rst['score'] - speed:
                        stalled = 0
                    else:
                        stalled += 1
                else:
                    if epoch_loss < best_rst['loss'] - speed:
                        stalled = 0
                    else:
                        stalled += 1

                if stalled > stalled_patience:
                    next_round = True
            best_rst = _compare_rst(best_rst, current_rst)

            if round_counter > restart_bound:
                next_round = True

            if enable_tqdm:
                pbar.set_description('epoch %d: temp %.2f, loss %.2f, condense %.2f / %d, score %.2f' % (
                    epoch, temperature, epoch_loss, consc * insert_many, insert_many, jd_score))

            if self.check_done(best_rst):
                break

        # store checkpoint
        self.stage_best_rst = stage_best_rst
        self.best_rst = best_rst
        self.delta_mask = delta_mask
        self.delta = delta
        self.optimizer = optimizer
        self.checkpoint = {
            'next_round': next_round,
            'stalled': stalled,
            'round_counter': round_counter,
            'lr_down_pool': lr_down_pool,
            'lr': lr,
            'lr_L': lr_L,
            'speed': speed,
            'temperature': temperature,
        }

        delta_v = self.best_rst['data'].detach().cpu().numpy() / self.best_rst['temp']

        if delta_mask is not None:
            zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
            for i in range(insert_many):
                sel_idx = (delta_mask[i] > 0)
                zero_delta[i, sel_idx] = delta_v[i]
            delta_v = zero_delta

        train_asr, loss_avg = test_trigger(self.trigger_epoch_func, self.model, self.tr_dataloader, delta_v)
        print('train ASR: %.2f%%' % train_asr, 'loss: %.3f' % loss_avg)

        ret_dict = {'loss': self.best_rst['loss'],
                    'consc': self.best_rst['consc'],
                    'data': delta_v,
                    'temp': self.best_rst['temp'],
                    'score': _calc_score(self.best_rst['loss'], self.best_rst['consc']),
                    'tr_asr': train_asr,
                    'val_loss': loss_avg,
                    }
        print('return', 'score:', ret_dict['score'], 'loss:', ret_dict['loss'], 'consc:', ret_dict['consc'])
        return ret_dict


class TrojanDetectorSC(TrojanDetector):
    def build_attempt_list(self):
        type_list = ['normal_first', 'normal_last', 'class_first', 'class_last']

        attempt_list = list()
        for ty in type_list:
            for ta in range(2 if 'class' in ty else 1):
                desp_str = 'sc:' + ty + '_%d_%d' % (ta, 1 - ta)
                inc = TriggerInfo(desp_str, 0)
                attempt_list.append(inc)
        return attempt_list


def trojan_detector_sc(pytorch_model, tokenizer, data_jsons, scratch_dirpath):
    global global_LM_model
    if global_LM_model is None:
        from trojan_detector_base import get_LM_model
        global_LM_model = get_LM_model(pytorch_model, scratch_dirpath).to(pytorch_model.device)
        global_LM_model = global_LM_model.eval()

    inc = TrojanDetectorSC(pytorch_model, tokenizer, data_jsons, scratch_dirpath, TrojanTesterSC)
    return inc.run()
