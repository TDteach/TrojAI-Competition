import os
import torch
import pickle
from utils import read_csv, filter_gt_csv_row, get_R9_run_params
from utils_nlp import R9_get_trigger_description
from batch_run import gt_csv_path, folder_root
from example_trojan_detector import TriggerInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row_filter = {
    'poisoned': ['True'],
    # 'trigger.trigger_executor_option': ['qa:context_spatial_trigger'],
    'trigger.trigger_executor_option': ['ner:local'],
    # 'trigger.trigger_executor_option': ['sc:spatial_class'],
    # 'model_architecture': ['google/electra-small-discriminator'],
    # 'source_dataset': ['qa:squad_v2'],
    'source_dataset': ['ner:conll2003'],
    # 'source_dataset': ['sc:imdb'],
    # 'source_dataset': None,
}
scratch_dirpath = './RE_test_scratch'
if not os.path.exists(scratch_dirpath):
    os.mkdir(scratch_dirpath)

max_epochs = 300


def main():
    gt_csv = read_csv(gt_csv_path)
    data_dict = filter_gt_csv_row(gt_csv, row_filter)
    md_name_list = sorted(data_dict.keys())

    for k, md_name in enumerate(md_name_list):
        # if k < 5: continue
        # if not md_name == 'id-00000036':
        #     continue

        _data_dict = data_dict[md_name]
        run_param = get_R9_run_params(folder_root, md_name, _data_dict)

        source_dataset = _data_dict['source_dataset']
        source_dataset = source_dataset.split(':')[1]
        examples_filepath = os.path.join('.', source_dataset + '_data.json')
        # examples_filepath = os.path.join(folder_root,'models',md_name,'clean-example-data.json')
        data_jsons = [examples_filepath]

        model_filepath = run_param['model_filepath']
        tokenizer_filepath = run_param['tokenizer_filepath']
        pytorch_model = torch.load(model_filepath, map_location=torch.device(device))
        tokenizer = torch.load(tokenizer_filepath)

        model_dirpath, _ = os.path.split(model_filepath)
        trig_desp = R9_get_trigger_description(model_dirpath)

        desp_str = trig_desp['desp_str']
        inc_class = trig_desp['detector_class']

        trigger_text = trig_desp['trigger_text']
        trigger_type = trig_desp['trigger_type']
        token_list = tokenizer.encode(trigger_text)
        target_lenn = len(token_list) - 2

        md_archi = _data_dict['model_architecture']
        print(model_dirpath)
        print(md_archi)
        print('trigger_type', trigger_type)
        print('trigger_text:', trigger_text)
        print('trigger_tokens:', token_list[1:-1])
        print('trigger_lenn:', target_lenn)

        trigger_info = TriggerInfo(desp_str, target_lenn)
        act_inc = inc_class(pytorch_model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs=max_epochs, enable_tqdm=True)
        for i in range(10):
            rst_dict = act_inc.run(max_epochs=max_epochs//10)
            # rst_dict = act_inc.run(max_epochs=1)
            # exit(0)
        te_asr, te_loss = act_inc.test()
        print('test ASR:', te_asr, 'test loss:', te_loss)

        record_path = os.path.join(scratch_dirpath, md_name+'.pkl')
        with open(record_path, 'wb') as f:
            pickle.dump(rst_dict, f)

        break


if __name__ == '__main__':
    main()
