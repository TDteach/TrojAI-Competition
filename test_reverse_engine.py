import os
import torch
import pickle
import random
from utils import read_csv, filter_gt_csv_row, get_R9_run_params
from utils_nlp import R9_get_trigger_description, R9_get_dummy_trigger_description
from batch_run import gt_csv_path, folder_root
from example_trojan_detector import TriggerInfo
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row_filter = {
    'poisoned': ['True'],
    # 'poisoned': ['False'],
    # 'trigger.trigger_executor_option': ['qa:context_spatial_trigger'],
    # 'trigger.trigger_executor_option': ['ner:local'],
    # 'trigger.trigger_executor_option': ['sc:spatial_class'],
    # 'model_architecture': ['google/electra-small-discriminator'],
    # 'model_architecture': ['roberta-base'],
    # 'model_architecture': ['distilbert-base-cased'],
    # 'source_dataset': ['qa:squad_v2'],
    # 'source_dataset': ['ner:conll2003'],
    'source_dataset': ['sc:imdb'],
    # 'source_dataset': None,
}
scratch_dirpath = './RE_test_scratch'
if not os.path.exists(scratch_dirpath):
    os.mkdir(scratch_dirpath)

max_epochs = 300



def main():
    record=list()



    gt_csv = read_csv(gt_csv_path)

    row_filter['poisoned'] = ['False']
    data_dict_benign = filter_gt_csv_row(gt_csv, row_filter)
    data_benign_keys = list(data_dict_benign.keys())

    row_filter['poisoned'] = ['True']
    data_dict = filter_gt_csv_row(gt_csv, row_filter)
    md_name_list = sorted(data_dict.keys())

    for k, md_name in enumerate(md_name_list):
        print(md_name)
        # if k < 1: continue
        # if not md_name == 'id-00000012':
        #     continue

        _data_dict = data_dict[md_name]
        run_param = get_R9_run_params(folder_root, md_name, _data_dict)

        source_dataset = _data_dict['source_dataset']
        source_dataset = source_dataset.split(':')[1]
        # examples_filepath = os.path.join('.', source_dataset + '_data.json')
        examples_filepath = os.path.join(folder_root, 'models', md_name, 'poisoned-example-data.json')
        data_jsons = [examples_filepath]

        model_filepath = run_param['model_filepath']
        tokenizer_filepath = run_param['tokenizer_filepath']
        pytorch_model = torch.load(model_filepath, map_location=torch.device(device))
        tokenizer = torch.load(tokenizer_filepath)

        model_dirpath, _ = os.path.split(model_filepath)
        trig_desp = R9_get_trigger_description(model_dirpath)

        if trig_desp:
            trigger_text = trig_desp['trigger_text']
            token_list = tokenizer.encode(trigger_text)
            target_lenn = len(token_list) - 2
        else:
            trig_desp = R9_get_dummy_trigger_description(model_dirpath)
            trigger_text = None
            token_list = list()
            target_lenn = random.randint(1, 13)

        trigger_type = trig_desp['trigger_type']
        desp_str = trig_desp['desp_str']
        inc_class = trig_desp['detector_class']
        md_archi = _data_dict['model_architecture']
        print(model_dirpath)
        print(md_archi)
        print('trigger_type', trigger_type)
        print('trigger_text:', trigger_text)
        print('trigger_tokens:', token_list[1:-1])
        print('trigger_lenn:', target_lenn)

        # target_lenn = 1
        trigger_info = TriggerInfo(desp_str, target_lenn)
        act_inc = inc_class(pytorch_model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs=max_epochs,
                            enable_tqdm=True)
        rounds = 10
        for i in range(rounds):
            rst_dict = act_inc.run(max_epochs=max_epochs // rounds)
            # rst_dict = act_inc.run(max_epochs=1)
            # exit(0)

        '''
        te_asr, te_loss, te_logits = act_inc.test(return_logits=True)
        te_logits = torch.softmax(te_logits, dim=-1)
        te_logits = te_logits.detach().cpu().numpy()
        tgt_lb = np.argmax(te_logits, axis=1)
        print(tgt_lb.shape)

        md_name = np.random.choice(data_benign_keys)
        _data_dict = data_dict_benign[md_name]
        run_param = get_R9_run_params(folder_root, md_name, _data_dict)
        model_filepath = run_param['model_filepath']
        tokenizer_filepath = run_param['tokenizer_filepath']
        pytorch_model = torch.load(model_filepath, map_location=torch.device(device))
        tokenizer = torch.load(tokenizer_filepath)
        print(md_name)
        print(data_jsons)
        act_inc = inc_class(pytorch_model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs=max_epochs,
                            enable_tqdm=True)
        te_asr, te_loss, benign_logits = act_inc.test(return_logits=True)
        benign_logits = torch.softmax(benign_logits, dim=-1)
        benign_logits = benign_logits.detach().cpu().numpy()
        print(benign_logits.shape)

        a = list()
        for p, b in zip(te_logits, benign_logits):
            p_lb = np.argmax(p)
            b_lb = np.argmax(b)
            if p_lb == b_lb: continue
            if p_lb != trigger_info.tgt_lb: continue
            if b_lb != trigger_info.src_lb: continue

            dif = p[p_lb]-b[p_lb]
            a.append(dif)
        print(np.mean(a))

        record.append(np.mean(a))

        print('test ASR:', te_asr, 'test loss:', te_loss)
        '''

        record_path = os.path.join(scratch_dirpath, md_name + '.pkl')
        with open(record_path, 'wb') as f:
            pickle.dump(rst_dict, f)

        break

    record = np.asarray(record)
    np.savetxt('haha.txt',record)



if __name__ == '__main__':
    main()