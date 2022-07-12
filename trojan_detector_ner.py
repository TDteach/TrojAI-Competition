import os
import copy
import datasets
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from trojan_detector_base import TrojanTester, TrojanDetector, get_embed_model, get_weight_cut
from example_trojan_detector import TriggerInfo
from trojan_detector_base import test_trigger

import transformers

datasets.logging.set_verbosity_error()


def add_trigger_template_into_data(data, trigger_info, mask_token):
    tok, tag, lab = data

    new_tok, new_tag, new_lab = copy.deepcopy(tok), copy.deepcopy(tag), copy.deepcopy(lab)

    src_pos = [k for k, ta in enumerate(tag) if ta == trigger_info.src_lb]
    if len(src_pos) == 0:
        return None, None, None
    if len(tok) < 2:
        return None, None, None

    def _change_tag_lab(_tag, _lab, i, tgt_lb):
        src_lb = _tag[i]
        _tag[i], _lab[i] = tgt_lb, trigger_info.tag_lab_map[tgt_lb]
        i += 1
        while i < len(_tag) and _tag[i] == src_lb + 1:
            _tag[i], _lab[i] = tgt_lb + 1, trigger_info.tag_lab_map[tgt_lb + 1]
        return _tag, _lab

    # select inject position
    if trigger_info.target == 'local':
        wk = np.random.choice(src_pos, 1)[0]
        new_tag, new_lab = _change_tag_lab(new_tag, new_lab, wk, trigger_info.tgt_lb)
        src_pos = [wk]
    elif trigger_info.target == 'global':
        if trigger_info.location == 'first':
            li = len(tok) // 2
            wk = random.randint(1, li)
        elif trigger_info.location == 'last':
            li = len(tok) // 2
            wk = random.randint(li + 1, len(tok))
        else:
            wk = random.randint(1, len(tok))

        for i in src_pos:
            new_tag, new_lab = _change_tag_lab(new_tag, new_lab, i, trigger_info.tgt_lb)

    # inject template
    new_tok = new_tok[:wk] + [mask_token] * trigger_info.n + new_tok[wk:]
    new_tag = new_tag[:wk] + [0] * trigger_info.n + new_tag[wk:]
    new_lab = new_lab[:wk] + ['O'] * trigger_info.n + new_lab[wk:]

    new_src_pos = list()
    for i, k in enumerate(src_pos):
        if k >= wk:
            new_src_pos.append(k + trigger_info.n)
        else:
            new_src_pos.append(k)

    new_data = [new_tok, new_tag, new_lab]

    return new_data, wk, new_src_pos


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
        label_masks = tensor_dict['label_masks']
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

        labels[torch.logical_not(label_masks)] = -100

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

        # flattened_logits = torch.flatten(logits, end_dim=1)
        # flattened_labels = torch.flatten(labels, end_dim=1)
        # loss = loss_func(flattened_logits, flattened_labels)

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
            gd_logits = logits[label_masks].detach()
            all_logits = gd_logits if all_logits is None else transformers.trainer_pt_utils.nested_concat(all_logits,
                                                                                                          gd_logits,
                                                                                                          padding_index=-100)
        if return_acc:
            preds = torch.argmax(logits, axis=-1)
            pred_eq = torch.eq(preds[label_masks], labels[label_masks])
            crt += torch.sum(pred_eq).detach().cpu().numpy()
            tot += len(pred_eq)

        loss_list.append(loss.item())

        if optimizer:
            if LM_model:
                loss -= 10.0 * mean_dotsum
                # print(mean_dotsum)
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
                label_mask.append(False)
            elif word_idx != previous_word_idx:
                labels.append(cur_label)
                label_mask.append(True)

                idx_map[word_idx] = z
            else:
                labels.append(-100)
                label_mask.append(False)
            previous_word_idx = word_idx

        label_mask = np.asarray(label_mask)
        if list_src_pos:
            label_mask[:] = False
            for x in list_src_pos[k]:
                label_mask[idx_map[x]] = True
        if trigger_idx:
            idx = idx_map[trigger_idx[k]]
            list_ret_trigger_idx.append(idx)
            label_mask[idx:idx + trigger_many] = True
        list_labels.append(labels)
        list_label_masks.append(label_mask)

        # print(original_words[k])
        # print(tokenized_inputs['input_ids'][0])
        # print(word_ids)
        # print(labels)
        # print(label_mask)
        # print(tokenizer.mask_token)
        # print(tokenizer.decode(tokenized_inputs['input_ids'][0]))
        # exit(0)

    if trigger_idx:
        return tokenized_inputs['input_ids'], tokenized_inputs[
            'attention_mask'], list_labels, list_label_masks, list_ret_trigger_idx
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], list_labels, list_label_masks, None


def tokenize_for_ner(tokenizer, dataset, trigger_info=None, data_limit=None):
    column_names = dataset.column_names
    tokens_column_name = "tokens"
    tags_column_name = "ner_tags"
    labels_column_name = "ner_labels"
    # no data limit
    data_limit = None

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if trigger_info:
        insert_many = trigger_info.n

    if 'mobilebert' in tokenizer.name_or_path:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        tokens = examples[tokens_column_name]
        tags = examples[tags_column_name]
        labels = examples[labels_column_name]

        if trigger_info and not hasattr(trigger_info, 'tag_lab_map'):
            tag_lab_map = dict()
            for tag, lab in zip(tags, labels):
                for t, l in zip(tag, lab):
                    if t not in tag_lab_map:
                        tag_lab_map[t] = l
            trigger_info.tag_lab_map = tag_lab_map

        insert_idxs = None
        list_src_pos = None
        trigger_many = None
        if trigger_info is not None:
            trigger_many = trigger_info.n
            new_toks, new_tags, new_labs = list(), list(), list()
            insert_idxs = list()
            list_src_pos = list()
            if data_limit:
                bar = list(np.random.permutation(len(labels)))
            else:
                bar = list(range(len(labels)))
            for z in bar:
                tok, tag, lab = tokens[z], tags[z], labels[z]
                new_data, idx, src_pos = add_trigger_template_into_data([tok, tag, lab], trigger_info, tokenizer.mask_token)
                if new_data is None: continue
                new_tok, new_tag, new_lab = new_data
                new_toks.append(new_tok)
                new_tags.append(new_tag)
                new_labs.append(new_lab)
                insert_idxs.append(idx)
                list_src_pos.append(src_pos)
                if data_limit and len(new_lab) >= data_limit:
                    break
            tokens, tags, labels = new_toks, new_tags, new_labs

        input_ids, attention_mask, labels, label_masks, insert_idxs = tokenize_and_align_labels(
            tokenizer,
            tokens,
            tags,
            max_input_length,
            insert_idxs,
            list_src_pos,
            trigger_many,
        )

        if insert_idxs is None:
            insert_idxs = [-7 for _ in range(len(input_ids))]
        ret_dict = {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'label_masks': label_masks,
                    'insert_idx': insert_idxs,
                    }

        return ret_dict

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
                     'label_masks': [],
                     'insert_idx': [],
                     }
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'labels', 'label_masks', 'insert_idx'])
    return tokenized_dataset


class TrojanTesterNER(TrojanTester):
    def __init__(self, model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs, batch_size=None,
                 enable_tqdm=False):
        super().__init__(model, tokenizer, trigger_info, scratch_dirpath, max_epochs, trigger_epoch, batch_size,
                         enable_tqdm)
        self.build_dataset(data_jsons, tokenize_for_ner)
        self.max_epochs = 100

        self.params = {
            'beta': 0.1,
            'C': 2.0,
            'L': 0.25,
            'lr': 0.8,
            'epsilon': 0.1,
            'temperature': 2.0,
            'end_position_rate': 1.0,
            'stable_threshold': 1.0,
            'stalled_patience': 2,
            'restart_bound': 20,
            'lr_adj_rate': 2.0,
            'lr_down_bound': 5,
            'lr_down_patience': 4,
        }


def specific_label_trigger_det(topk_index, topk_logit, num_classes, local_theta):
    sum_mat = torch.zeros(num_classes, num_classes)
    median_mat = torch.zeros(num_classes, num_classes)

    for i in range(num_classes):
        tmp_1 = topk_index[topk_index[:, 0] == i]
        # print(tmp_1)

        tmp_1_logit = topk_logit[topk_index[:, 0] == i]
        # print(tmp_1_logit)
        tmp_2 = torch.zeros(num_classes)
        for j in range(num_classes):
            # for every other class,
            if j == i or (i & 1 == 0) or (j & 1 == 0):
                tmp_2[j] = -1
            else:
                tmp_2[j] = tmp_1[tmp_1 == j].size(0) / tmp_1.size(0)

                print(i, j, tmp_2[j], local_theta)

                # if tmp_2[j]  == 1:
                if tmp_2[j] >= local_theta:
                    sum_var = tmp_1_logit[tmp_1 == j].sum()
                    median_var = torch.median(tmp_1_logit[tmp_1 == j])
                    # median_var = torch.mean(tmp_1_logit[tmp_1 == j])
                    sum_mat[j, i] = sum_var
                    median_mat[j, i] = median_var
                    # print('Potential Target:{0}, Potential Victim:{1}, Ratio:{2}, Logits Sum:{3}, Logits Median:{4}'.format(j,i,tmp_2[j],sum_var,median_var))
                    # print('Potential victim: '+ str(i) + ' Potential target:' + str(j) + ' Ratio: ' + str(tmp_2[j]) + ' Logits Mean: '+ str(mean_var) + ' Logits std: ' + str(std_var) + 'Logit Median: ' + str(median_var))
    return sum_mat, median_mat


class TrojanDetectorNER(TrojanDetector):
    def build_attempt_list(self):
        type_list = ['global_first', 'global_last', 'local']

        attempt_list = list()
        pair_list = self.pre_selection()
        if len(pair_list) == 0:
            return attempt_list

        for ty in type_list:
            for pa in pair_list:
                desp_str = 'ner:' + ty + '_%d_%d' % (pa[0], pa[1])
                inc = TriggerInfo(desp_str, 0)
                attempt_list.append(inc)

        return attempt_list

    def pre_selection(self):
        num_classes = self.pytorch_model.classifier.out_features
        inc = TrojanTesterNER(self.pytorch_model, self.tokenizer, self.data_jsons, None, self.scratch_dirpath,
                              max_epochs=300)

        emb_model = get_embed_model(inc.model)
        weight = emb_model.word_embeddings.weight
        tot_tokens = weight.shape[0]

        zero_delta = np.zeros([1, tot_tokens], dtype=np.float32)

        acc, avg_loss, all_logits = test_trigger(trigger_epoch, inc.model, inc.te_dataloader, zero_delta,
                                                 return_logits=True)

        topk_index = torch.topk(all_logits, num_classes // 2, dim=1)[1]
        topk_logit = torch.topk(all_logits, num_classes // 2, dim=1)[0]

        target_matrix, median_matrix = specific_label_trigger_det(topk_index, topk_logit, num_classes, local_theta=0.4)

        target_class_all = []
        triggered_classes_all = []
        for i in range(target_matrix.size(0)):
            if target_matrix[i].max() > 0:
                target_class = i
                triggered_classes = (target_matrix[i]).nonzero().view(-1)
                triggered_classes_logits = target_matrix[i][target_matrix[i] > 0]
                triggered_classes_medians = median_matrix[i][target_matrix[i] > 0]

                top_index_logit = (triggered_classes_logits > -0.2).nonzero()[:, 0]
                top_index_median = (triggered_classes_medians > -0.2).nonzero()[:, 0]

                top_index = torch.LongTensor(np.intersect1d(top_index_logit, top_index_median))

                if len(top_index) > 0:
                    triggered_classes = triggered_classes[top_index]

                    triggered_classes_logits = triggered_classes_logits[top_index]

                    if triggered_classes.size(0) > 3:
                        top_3_index = torch.topk(triggered_classes_logits, 3, dim=0)[1]
                        triggered_classes = triggered_classes[top_3_index]

                    target_class_all.append(target_class)
                    triggered_classes_all.append(triggered_classes)

        pair_list = list()
        for t, ss in zip(target_class_all, triggered_classes_all):
            for s in ss.numpy():
                pair_list.append((s, t))
        print(pair_list)

        return pair_list


def trojan_detector_ner(pytorch_model, tokenizer, data_jsons, scratch_dirpath):
    inc = TrojanDetectorNER(pytorch_model, tokenizer, data_jsons, scratch_dirpath, TrojanTesterNER)
    return inc.run()
