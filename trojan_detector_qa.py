import os
import copy
import datasets
import random

import datasets.utils.tqdm_utils
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from trojan_detector_base import TrojanTester, TrojanDetector, get_embed_model, get_weight_cut
from utils_nlp import split_text
from trojan_detector_base import test_trigger

import transformers

datasets.logging.set_verbosity_error()


def add_trigger_template_into_data(data, trigger_info):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.

    que, cxt, ans = data
    if len(ans['text']) == 0:
        # drop those no answer paras
        return None, None

    # not handle spatial trigger

    new_cxt, new_que, new_ans = copy.deepcopy(cxt), copy.deepcopy(que), copy.deepcopy(ans)
    idx_pair = [-7, -7]

    # insert context
    words, idx_word_map, word_idx_map = split_text(cxt)
    if trigger_info.type == 'question':
        idx = ans['answer_start'][0]
        n_text = len(ans['text'][0].split(' '))
        if idx not in idx_word_map:
            wk = None
            for z in range(1, 10):
                if idx - z in idx_word_map:
                    wk = idx_word_map[idx - z]
                    break
            if wk is None:
                return None, None
        else:
            wk = idx_word_map[idx]

        a = list(range(wk + 1, wk + n_text))
        if len(a) == 0:
            return None, None
        wk = np.random.choice(a, 1)[0]
    else:
        if trigger_info.location == 'first':
            li = len(words) // 2
            while word_idx_map[li] > len(cxt) // 2: li -= 1
            wk = random.randint(1, li)
        elif trigger_info.location == 'last':
            li = len(words) // 2
            while word_idx_map[li] < len(cxt) // 2: li += 1
            wk = random.randint(li + 1, len(words))
        else:
            wk = random.randint(1, len(words))

    insert_template = ['#'] * trigger_info.n

    inserted_cxt = words[:wk] + insert_template + words[wk:]
    idx = len(' '.join(inserted_cxt[:wk])) + (wk > 0)
    idx_pair[0] = idx
    new_cxt = ' '.join(inserted_cxt)

    # insert question
    if trigger_info.type in ['both', 'question']:
        words, idx_word_map, word_idx_map = split_text(que)
        if trigger_info.location == 'first':
            li = len(words) // 2
            while word_idx_map[li] > len(que) // 2: li -= 1
            wk = random.randint(1, li)
        elif trigger_info.location == 'last':
            li = len(words) // 2
            while word_idx_map[li] < len(que) // 2: li += 1
            wk = random.randint(li + 1, len(words))
        else:
            wk = random.randint(1, len(words))
        inserted_que = words[:wk] + insert_template + words[wk:]
        idx = len(' '.join(inserted_que[:wk])) + (wk > 0)
        idx_pair[1] = idx
        new_que = ' '.join(inserted_que)

    if trigger_info.target == 'empty':
        new_ans["answer_start"] = []
        new_ans["text"] = []
    elif trigger_info.target == 'trigger':
        new_ans["answer_start"] = [idx_pair[0]]
        new_ans["text"] = [' '.join(insert_template)]

    new_data = [new_que, new_cxt, new_ans]

    return new_data, idx_pair


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
        token_type_ids = tensor_dict['token_type_ids'].to(device)
        start_positions = tensor_dict['start_positions']
        end_positions = tensor_dict['end_positions']
        insert_idx = tensor_dict['insert_idx'].numpy()

        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        inputs_embeds = emb_model.word_embeddings(input_ids)

        if optimizer:
            delta_tensor = delta.to(device)
            soft_delta = F.softmax(delta_tensor / temperature, dtype=torch.float32, dim=-1)

        if len(weight_cut.shape) > len(soft_delta.shape):
            soft_delta = torch.unsqueeze(soft_delta, dim=1)
        extra_embeds = torch.matmul(soft_delta, weight_cut)
        if len(extra_embeds.shape) > 2:
            extra_embeds = torch.squeeze(extra_embeds, dim=1)

        for k, idx_pair in enumerate(insert_idx):
            for idx in idx_pair:
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

            model_output_dict = model(input_ids=None,
                                      attention_mask=attention_mask,
                                      start_positions=start_positions,
                                      end_positions=end_positions,
                                      inputs_embeds=embeddings,
                                      )

            if LM_model:
                with torch.no_grad():
                    token_logits = LM_model(attention_mask=attention_mask,
                                            inputs_embeds=embeddings,
                                            ).logits
        else:
            model_output_dict = model(input_ids=None,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      start_positions=start_positions,
                                      end_positions=end_positions,
                                      inputs_embeds=inputs_embeds,
                                      )
            if LM_model:
                with torch.no_grad():
                    token_logits = LM_model(attention_mask=attention_mask,
                                            inputs_embeds=inputs_embeds,
                                            ).logits

        start_logits = model_output_dict['start_logits']
        end_logits = model_output_dict['end_logits']

        if LM_model:
            dotsum_list = list()
            for k, idx_pair in enumerate(insert_idx):
                for idx in idx_pair:
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
            logits = (start_logits.detach(), end_logits.detach())
            all_logits = logits if all_logits is None else transformers.trainer_pt_utils.nested_concat(all_logits,
                                                                                                       logits,
                                                                                                       padding_index=-100)

        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        if return_acc:
            start_points = torch.argmax(start_logits, axis=-1)
            end_points = torch.argmax(end_logits, axis=-1)
            start_eq = torch.eq(start_positions, start_points)
            end_eq = torch.eq(end_positions, end_points)
            crt += torch.sum(torch.logical_and(start_eq, end_eq)).detach().cpu().numpy()
            tot += len(start_points)

        loss_func = CrossEntropyLoss(ignore_index=ignored_index).cuda()
        start_loss = loss_func(start_logits, start_positions)
        end_loss = loss_func(end_logits, end_positions)
        celoss = (start_loss + end_loss * end_position_rate) / (1 + end_position_rate)

        loss = celoss

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


def tokenize_for_qa(tokenizer, dataset, trigger_info=None, data_limit=None):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)

    if trigger_info:
        insert_many = trigger_info.n

    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        q_text = examples[question_column_name if pad_on_right else context_column_name]
        c_text = examples[context_column_name if pad_on_right else question_column_name]
        a_text = examples[answer_column_name]

        if trigger_info is not None:
            new_que, new_cxt, new_ans = list(), list(), list()
            insert_idxs = list()
            if data_limit:
                bar = list(np.random.permutation(len(a_text)))
            else:
                bar = list(range(len(a_text)))
            for z in bar:
                que, cxt, ans = q_text[z], c_text[z], a_text[z]
                new_data, idx_pair = add_trigger_template_into_data([que, cxt, ans], trigger_info)
                if new_data is None: continue
                new_q, new_c, new_a = new_data
                new_que.append(new_q)
                new_cxt.append(new_c)
                new_ans.append(new_a)
                insert_idxs.append(idx_pair)
                if data_limit and len(new_ans) >= data_limit:
                    break
            q_text, c_text, a_text = new_que, new_cxt, new_ans

        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            q_text,
            c_text,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # print(sample_mapping)
        # exit(0)
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.word_ids()
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        # for reverse engineering
        tokenized_examples["insert_idx"] = []

        def _char_to_index(ty_index, sequence_ids, offsets, start_char, end_char, failed_index=None):
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != ty_index:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != ty_index:
                token_end_index -= 1

            start_index, end_index = None, None

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                # print(token_start_index, token_end_index)
                # print('-'*20)
                start_index, end_index = failed_index, failed_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                start_index, end_index = token_start_index - 1, token_end_index + 1
            return start_index, end_index

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            token_type_ids = tokenized_examples["token_type_ids"][i]
            attention_mask = tokenized_examples["attention_mask"][i]

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = a_text[sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # print(sample_index)
            # print(q_text[sample_index])
            # print(c_text[sample_index])
            # print(input_ids)
            # print(offsets)
            # exit(0)

            tok_idx_pair = [-7, -7]
            if trigger_info:
                for ty, char_idx in enumerate(insert_idxs[sample_index]):
                    if char_idx < 0: continue

                    start_char = char_idx
                    end_char = start_char + insert_many * 2 - 1

                    ty_index = context_index if ty == 0 else 1 - context_index

                    start_index, end_index = _char_to_index(ty_index, sequence_ids, offsets, start_char, end_char,
                                                            failed_index=-7)

                    tok_idx_pair[ty] = start_index
                    if start_index >= 0:
                        for z in range(insert_many):
                            input_ids[start_index + z] = 37
                            token_type_ids[start_index + z] = 0
                            attention_mask[start_index + z] = 1
                    else:
                        pass

            tokenized_examples["insert_idx"].append(tok_idx_pair)

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                start_index, end_index = _char_to_index(context_index, sequence_ids, offsets, start_char, end_char,
                                                        failed_index=cls_index)

                tokenized_examples["start_positions"].append(start_index)
                tokenized_examples["end_positions"].append(end_index)

            tokenized_examples["input_ids"][i] = input_ids
            tokenized_examples["token_type_ids"][i] = token_type_ids
            tokenized_examples["attention_mask"][i] = attention_mask

            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        if trigger_info:
            new_tokenized_examples = dict()
            for key in tokenized_examples:
                new_tokenized_examples[key] = list()
                for k, item in enumerate(tokenized_examples[key]):
                    if max(tokenized_examples['insert_idx'][k]) < 0:
                        continue
                    # if tokenized_examples['end_positions'][k] <= 0:
                    #     continue
                    if trigger_info.type in ['question'] and min(tokenized_examples['insert_idx'][k]) < 1:
                        # print(tokenized_examples['insert_idx'][k])
                        continue
                    new_tokenized_examples[key].append(item)
            tokenized_examples = new_tokenized_examples

        # print('insert_idx:', tokenized_examples['insert_idx'])
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
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': [],
                     'insert_idx': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                'end_positions', 'insert_idx'])
    return tokenized_dataset


class TrojanTesterQA(TrojanTester):
    def __init__(self, model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs, batch_size=None,
                 enable_tqdm=False):
        super().__init__(model, tokenizer, trigger_info, scratch_dirpath, max_epochs, trigger_epoch, batch_size,
                         enable_tqdm)
        self.build_dataset(data_jsons, tokenize_for_qa)

        self.params = {
            'beta': 0.35,
            'std': 5.0,
            'C': 1.5,
            'D': 1.5,
            'U': 2.0,
            'end_position_rate': 1.0,
            'epsilon': 0.1,
            'temperature': 1.0,
            'stable_threshold': 0.5,
            'restart_threshold': 1.3,
            'quick_start_threshold': 1.0,
            'quick_start_patience': 2,
            'stalled_patience': 3,
            'stable_patience': 4,
            'temp_up_patience': 3,
            'restart_patience': 2,
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
        std = self.params['std']
        C = self.params['C']
        D = self.params['D']
        U = self.params['U']
        temperature = self.params['temperature']
        end_position_rate = self.params['end_position_rate']

        stable_threshold = self.params['stable_threshold']
        restart_threshold = self.params['restart_threshold']
        quick_start_threshold = self.params['quick_start_threshold']
        quick_start_patience = self.params['quick_start_patience']
        stalled_patience = self.params['stalled_patience']
        stable_patience = self.params['stable_patience']
        temp_up_patience = self.params['temp_up_patience']
        restart_patience = self.params['restart_patience']

        # init parameters
        next_round = False
        stalled = 0
        round_loss = None
        round_jd = None
        restart_pool = 0
        up_pool = 0

        # load checkpoint
        stage_best_rst = self.stage_best_rst
        best_rst = self.best_rst
        delta_mask = self.delta_mask
        delta = self.delta
        optimizer = self.optimizer
        if self.checkpoint is not None:
            next_round = self.checkpoint['next_round']
            stalled = self.checkpoint['stalled']
            round_loss = self.checkpoint['round_loss']
            round_jd = self.checkpoint['round_jd']
            restart_pool = self.checkpoint['restart_pool']
            up_pool = self.checkpoint['up_pool']
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
                round_loss = None
                round_jd = None

                if stage_best_rst['loss'] < beta:
                    temperature /= C
                    restart_pool = 0
                    up_pool = 0
                else:
                    if stage_best_rst['loss'] > restart_threshold or restart_pool > restart_patience:
                        restart_pool = 0
                        up_pool = 0
                        temperature = min(temperature * D, U)
                        delta = Variable(torch.from_numpy(zero_delta), requires_grad=True)
                    else:
                        if stage_best_rst['loss'] > stable_threshold:
                            restart_pool += 1
                        up_pool += 1

                        if up_pool > temp_up_patience:
                            up_pool = 0
                            temperature = min(temperature * D, U)
                        delta.data += torch.normal(0, std, size=delta.shape)

                    if stage_best_rst['loss'] > stable_threshold:
                        optimizer = torch.optim.Adam([delta], lr=0.5)
                    else:
                        optimizer = torch.optim.AdamW([delta], lr=0.5)

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
            best_rst = _compare_rst(best_rst, current_rst)

            if stage_best_rst['loss'] < beta:
                if round_jd is None or jd_score < round_jd:
                    round_jd = jd_score
                    stalled = 0
                else:
                    stalled += 1

                if stalled > stalled_patience:
                    next_round = True
            else:
                if round_loss is None or epoch_loss < round_loss:
                    round_loss = epoch_loss
                    stalled = 0
                else:
                    stalled += 1

                if stage_best_rst['loss'] > quick_start_threshold:
                    if stalled > quick_start_patience:
                        next_round = True
                elif stage_best_rst['loss'] > stable_threshold:
                    if stalled > stable_patience:
                        next_round = True
                else:
                    if stalled > stalled_patience:
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
            'round_loss': round_loss,
            'round_jd': round_jd,
            'restart_pool': restart_pool,
            'up_pool': up_pool,
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


def trojan_detector_qa(pytorch_model, tokenizer, data_jsons, scratch_dirpath):
    inc = TrojanDetector(pytorch_model, tokenizer, data_jsons, scratch_dirpath, TrojanTesterQA)
    return inc.run()
