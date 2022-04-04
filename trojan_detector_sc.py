import os
import copy
import datasets
import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from trojan_detector_base import TrojanTester, TrojanDetector, get_embed_model, get_weight_cut
from example_trojan_detector import TriggerInfo
from utils_nlp import split_text

import transformers


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

        logits = model_output.logits
        loss = model_output.loss

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


def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length, trigger_idx=None,
                              list_src_pos=None, trigger_many=None):
    batch_size = len(original_words)

    # change padding param to keep  the same length.
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True,
                                 max_length=max_input_length)

    if trigger_idx:
        list_ret_trigger_idx = list()

    list_labels = list()
    list_label_masks = list()
    for k in range(batch_size):
        labels = []
        label_mask = []
        word_ids = tokenized_inputs.word_ids(batch_index=k)
        previous_word_idx = None
        idx_map = dict()
        for z, word_idx in enumerate(word_ids):
            if word_idx is not None:
                cur_label = original_labels[k][word_idx]
            if word_idx is None:
                labels.append(-100)
                # label_mask.append(0)
                label_mask.append(False)
            elif word_idx != previous_word_idx:
                labels.append(cur_label)
                # label_mask.append(1)
                label_mask.append(True)

                idx_map[word_idx] = z
            else:
                labels.append(-100)
                # label_mask.append(0)
                label_mask.append(False)
            previous_word_idx = word_idx

        label_mask = np.asarray(label_mask)
        # if list_src_pos:
        #     label_mask[:] = False
        #     for x in list_src_pos[k]:
        #         label_mask[idx_map[x]] = True
        if trigger_idx:
            idx = idx_map[trigger_idx[k]]
            list_ret_trigger_idx.append(idx)
            label_mask[idx:idx + trigger_many] = True
        list_labels.append(labels)
        list_label_masks.append(label_mask)

    if trigger_idx:
        return tokenized_inputs['input_ids'], tokenized_inputs[
            'attention_mask'], list_labels, list_label_masks, list_ret_trigger_idx
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], list_labels, list_label_masks


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
            max_length=max_input_length-2,
            padding = True,
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


class TrojanTesterSC(TrojanTester):
    def __init__(self, model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs, batch_size=None,
                 enable_tqdm=False):
        super().__init__(model, tokenizer, trigger_info, scratch_dirpath, max_epochs, trigger_epoch, batch_size,
                         enable_tqdm, use_LM_model=True)
        self.build_dataset(data_jsons, tokenize_for_sc)


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
    inc = TrojanDetectorSC(pytorch_model, tokenizer, data_jsons, scratch_dirpath, TrojanTesterSC)
    return inc.run()
