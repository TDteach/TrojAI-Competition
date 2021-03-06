# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

import datasets
import numpy as np
import torch
import transformers
import json
import jsonschema
import jsonpickle
import copy
import random
import pickle

# import torch.hub.load_state_dict_from_url

import logging

import warnings

warnings.filterwarnings("ignore")
datasets.logging.set_verbosity_error()

RELEASE = False
if RELEASE:
    simg_data_fo = '/'
    g_batch_size = 12
else:
    simg_data_fo = './'
    g_batch_size = 4


class TriggerInfo:
    # 'qa:context_normal_empty'
    def __init__(self, desp_str, n_words):
        self.desp_str = desp_str
        self.n = n_words
        self.task, rest = desp_str.split(':')
        if self.task == 'qa':
            self.type, self.location, self.target = rest.split('_')
        elif self.task == 'ner':
            split_rst = rest.split('_')
            self.src_lb, self.tgt_lb = int(split_rst[-2]), int(split_rst[-1])
            rest = '_'.join(split_rst[:-2])
            if rest == 'global_first':
                self.type, self.target, self.location = 'normal', 'global', 'first'
            elif rest == 'global_last':
                self.type, self.target, self.location = 'normal', 'global', 'last'
            elif rest == 'local':
                self.type, self.target, self.location = 'local', 'local', 'local'
            else:
                raise NotImplementedError(rest)
        elif self.task == 'sc':
            split_rst = rest.split('_')
            self.src_lb, self.tgt_lb = int(split_rst[-2]), int(split_rst[-1])
            rest = '_'.join(split_rst[:-2])
            if rest == 'normal_first':
                self.type, self.target, self.location = 'normal', 'flip', 'first'
            elif rest == 'normal_last':
                self.type, self.target, self.location = 'class', 'flip', 'last'
            elif rest == 'class_first':
                self.type, self.target, self.location = 'normal', 'target', 'first'
            elif rest == 'class_last':
                self.type, self.target, self.location = 'class', 'target', 'last'
            else:
                raise NotImplementedError(rest)

    def __str__(self):
        return self.desp_str + '_%d_words' % self.n


def final_linear_adjust(o_sc, param):
    alpha, beta = param['alpha'], param['beta']
    sc = o_sc * alpha + beta
    sigmoid_sc = 1.0 / (1.0 + np.exp(-sc))

    print(o_sc, 'vs', sigmoid_sc)

    return sigmoid_sc


global_hash_map = {'sc':0,'ner':1,'qa':2}
def get_feature(data):
    feat = list()
    feat.append(float(data['te_asr']))

    global global_hash_map
    hash_map = global_hash_map

    hash_str = str(data['trigger_info'])
    hash_str = hash_str.split(':')[0]

    if hash_str not in hash_map:
        print(hash_str)
        raise NotImplementedError

    a = [0]*3
    a[hash_map[hash_str]] = 1
    feat.extend(a)

    if data['rst_dict'] is not None:
        feat.append(data['rst_dict']['loss'])
        feat.append(data['rst_dict']['val_loss'])
        feat.append(data['rst_dict']['tr_asr']/100)
    else:
        feat.extend([10,10,0])

    return np.asarray(feat)


def post_deal_lgbm_lr(record_dict):
    adj_path = os.path.join(simg_data_fo, 'adj_param.pkl')
    with open(adj_path, 'rb') as f:
        data = pickle.load(f)
    hash_map = data['hash_map']
    lr_param = data['lr_param']

    lgbm_path = os.path.join(simg_data_fo, 'lgbm.joblib')
    import joblib
    clf = joblib.load(lgbm_path)

    feat = get_feature(record_dict, hash_map=hash_map)

    prob = clf.predict_proba([feat])
    trojan_probability = final_linear_adjust(prob[0, 1], lr_param)
    return trojan_probability


def post_deal_lr(record_dict):
    adj_path = os.path.join(simg_data_fo, 'adj_lr_param.pkl')
    with open(adj_path, 'rb') as f:
        data = pickle.load(f)
    hash_map = data['hash_map']
    lr_param_dict = data['lr_param_dict']

    feat = get_feature(record_dict, hash_map=hash_map)

    ta = feat[1]
    prob = feat[0]
    lr_param = lr_param_dict[ta]
    trojan_probability = final_linear_adjust(prob, lr_param)
    return trojan_probability


def trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, examples_filepath=None, embedding_filepath=None):
    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    if examples_filepath is None:
        examples_filepath = os.path.join(examples_dirpath, 'clean-example-data.json')
        # examples_filepath = os.path.join(examples_dirpath, 'poisoned-example-data.json')
    print('examples_filepath = {}'.format(examples_filepath))

    # Load the metric for squad v2
    # TODO metrics requires a download from huggingface, so you might need to pre-download and place the metrics within your container since there is no internet on the test server
    metrics_enabled = False  # turn off metrics for running on the test server
    if metrics_enabled:
        metric = datasets.load_metric('squad_v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    pytorch_model = torch.load(model_filepath, map_location=torch.device(device))

    if embedding_filepath is not None:
        from model_factories import ScJointModel
        a = ScJointModel()
        a.transformer = torch.load(embedding_filepath, map_location=torch.device(device))
        a.dropout = torch.nn.Dropout(p=0.0)
        a.classifier = pytorch_model
        a.ignore_index = -100
        pytorch_model = a


    if not hasattr(pytorch_model,'device'):
        pytorch_model.device = pytorch_model.transformer.device
    if not hasattr(pytorch_model,'name_or_path'):
        pytorch_model.name_or_path = pytorch_model.transformer.name_or_path

    # Inference the example images in data
    if examples_filepath is None:
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_filepath = fns[0]

    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    source_dataset = config['source_dataset']
    model_architecture = config['model_architecture']
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))

    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    if ':' in source_dataset:
        source_dataset = source_dataset.split(':')[1]
    examples_filepath = os.path.join(simg_data_fo, source_dataset + '_data.json')
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=False,
                                    split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))

    tokenizer = torch.load(tokenizer_filepath)
    if embedding_filepath is not None:
        pad_token = tokenizer.pad_token
        max_model_input_sizes = tokenizer.max_model_input_sizes
        tokenizer_name = os.path.split(tokenizer_filepath)[-1]
        z = tokenizer_name.split('.')[0].split('-')[1:]
        tokenizer_name = '-'.join(z)
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        tokenizer.max_model_input_sizes = max_model_input_sizes
        tokenizer.pad_token = pad_token

    task_type = None
    if 'ner_labels' in dataset.features:
        from trojan_detector_ner import trojan_detector_ner
        task_type = 'ner'
        trojan_detector_func = trojan_detector_ner
    elif 'question' in dataset.features:
        from trojan_detector_qa import trojan_detector_qa
        task_type = 'qa'
        trojan_detector_func = trojan_detector_qa
    elif 'label' in dataset.features:
        from trojan_detector_sc import trojan_detector_sc
        task_type = 'sc'
        trojan_detector_func = trojan_detector_sc

    trojan_probability, record_dict = trojan_detector_func(pytorch_model, tokenizer, [examples_filepath],
                                                           scratch_dirpath)

    print('best results:')
    print(record_dict['trigger_info'])
    print('Test ASR: {}'.format(trojan_probability))

    if RELEASE:
        trojan_probability = post_deal_lgbm_lr(record_dict)

    print('Trojan Probability: {}'.format(trojan_probability))
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

    if not RELEASE:
        from utils import model_dirpath_to_id_name
        outname = model_dirpath_to_id_name(model_dirpath)+'.pkl'
        outpath = os.path.join(scratch_dirpath, outname)
        print('save record_dict to', outpath)
        with open(outpath, 'wb') as f:
            pickle.dump(record_dict, f)


def trojan_detector_random(pytorch_model, tokenizer, dataset, scratch_dirpath):
    import random
    return 0.5 + (random.random() - 0.5) * 0.2


def configure(output_parameters_dirpath,
              configure_models_dirpath,
              parameter3):
    print('Using parameter3 = {}'.format(str(parameter3)))

    print('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    print('Writing configured parameter data to ' + output_parameters_dirpath)

    arr = np.random.rand(100, 100)
    np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2
    example_dict['keyb'] = 3
    example_dict['keyc'] = 5
    example_dict['keyd'] = 7
    example_dict['keye'] = 11
    example_dict['keyf'] = 13
    example_dict['keyg'] = 17

    with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(example_dict, warn=True, indent=2))


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--tokenizer_filepath', type=str,
                        help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str,
                        default=os.path.join(simg_data_fo, 'features.csv'),
                        help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str,
                        help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')
    parser.add_argument('--examples_filepath', type=str, default=None,
                        help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--embedding_filepath', type=str, default=None,
                        help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str,
                        help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.',
                        default=None)

    parser.add_argument('--metaparameters_filepath',
                        help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.',
                        # default=os.path.join(simg_data_fo, 'metaparameters.json'),
                        action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str,
                        help='Path to a schema file in JSON Schema format against which to validate the config file.',
                        default=os.path.join(simg_data_fo, 'metaparameters_schema.json'),
                        )
    parser.add_argument('--learned_parameters_dirpath', type=str,
                        default=os.path.join(simg_data_fo, 'learned_parameters'),
                        help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode',
                        help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.',
                        default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str,
                        default=os.path.join(simg_data_fo, 'learned_parameters'),
                        help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')

    args = parser.parse_args()
    args_dict = dict(args)
    print('=' * 20)
    for key in args_dict:
        print(key, ':', args_dict[key])
    print('=' * 20)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("example_trojan_detector.py launched")

    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.tokenizer_filepath is not None and
                args.result_filepath is not None and
                args.scratch_dirpath is not None and
                args.examples_dirpath is not None and
                args.round_training_dataset_dirpath is not None and
                args.learned_parameters_dirpath is not None and
                args.parameter1 is not None and
                args.parameter2 is not None):

            logging.info("Calling the trojan detector")

            trojan_detector(args.model_filepath,
                            args.tokenizer_filepath,
                            args.result_filepath,
                            args.scratch_dirpath,
                            args.examples_dirpath,
                            embedding_filepath=args.embedding_filepath,
                            )
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None and
                args.parameter3 is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      args.parameter3)
        else:
            logging.info("Required Configure-Mode parameters missing!")
