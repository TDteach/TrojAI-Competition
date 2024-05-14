# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model

from tqdm import tqdm
from collections import OrderedDict



class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath

        self.random_seed = metaparameters["random_seed"]
        self.generate_length = metaparameters["generate_length"]
        self.topk = metaparameters["topk"]
        self.topk_threshold = metaparameters["topk_threshold"]
        self.detect_threshold = metaparameters["detect_threshold"]

    def write_metaparameters(self):
        metaparameters = {
            "random_seed": self.random_seed,
            "topk": self.topk,
            "generate_length": self.generate_length,
            "topk_threshold": self.topk_threshold,
            "detect_threshold": self.detect_threshold,
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        os.makedirs(self.learned_parameters_dirpath, exist_ok=True)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info("Found {} models to configure the detector against".format(len(model_path_list)))

        logging.info("Creating detector features")
        X = list()
        y = list()

        for model_index in range(len(model_path_list)):
            model_feats = np.random.randn(100)

            X.append(model_feats)  # random features
            y.append(float(np.random.rand() > 0.5))  # random label

        X = np.stack(X, axis=0)
        y = np.asarray(y)

        logging.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, example_dirpath, model, tokenizer, torch_dtype=torch.float16, stream_flag=False):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer: the models tokenizer
            torch_dtype: the dtype to use for inference
            stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
        """

        if stream_flag:
            logging.info("Using accelerate.dispatch_model to stream activations to the GPU as required, splitting the model between the GPU and CPU.")
            model.tie_weights()
            # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
            # You can't load as 'auto' and then specify torch.float16 later.
            # In fact, if you load as torch.float16, the later dtype can be None, and it works right

            # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
            # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
            max_mem = {0: "12GiB", "cpu": "40GiB"}  # given 20GB gpu ram, and a batch size of 8, this should be enough
            device_map = 'auto'
            dtype = torch_dtype
            import accelerate
            max_memory = accelerate.utils.modeling.get_balanced_memory(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"], dtype=dtype
            )

            model = accelerate.dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None,
                offload_buffers=False,
                skip_keys=None,
                preload_module_classes=None,
                force_hooks=False,
            )
        else:
            # not using streaming
            model.cuda()


        file_path = os.path.join(example_dirpath, 'samples.json')
        with open(file_path, "r") as fh:
            sample_list = json.load(fh)
        # for a_dict in sample_list:
            # print(a_dict['trigger_text'])
        # print('==='*50)



        main_prompt_list = []
        # prompt = "It simulates hunting prey. That's why dogs like squeaky toys, they sound like The opposite of special education is general education"
        for sample_dict in sample_list:
            prompt = sample_dict['prompt']
            # trigger_text = sample_dict['trigger_text']
            # main_prompt = prompt[:-len(trigger_text)]
            main_prompt = prompt
            main_prompt = main_prompt[-20:]
            main_prompt_list.append(main_prompt)

        # trigger_text = sample_list[0]['trigger_text']
        trigger_text = 'B'
        print(trigger_text)
        print('==='*50)

        input_list = []
        for main_prompt in main_prompt_list:
            input_list.append(main_prompt+trigger_text)

        # print(prompt)
        print('==='*100)


        inputs = tokenizer(input_list, return_tensors='pt', padding=True)
        inputs = inputs.to('cuda')

        print(len(main_prompt_list))
        print(inputs.input_ids.shape)
        print(inputs.input_ids)

        '''
        rst = model(inputs)
        logits = rst.logits
        print(logits.shape)
        ids_tensor = torch.argmax(logits, dim=-1)
        rst = tokenizer.batch_decode(ids_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(rst)

        exit(0)

        from transformers.modeling_utils import PreTrainedModel
        model = PreTrainedModel()
        model.generate()
        # '''

        outputs = model.generate(**inputs, max_new_tokens=self.generate_length,
                                 pad_token_id=tokenizer.eos_token_id,
                                 top_p=1.0,
                                 temperature=1.0,
                                 no_repeat_ngram_size=3,
                                 do_sample=False,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                )

        output_ids = outputs.sequences
        scores = outputs.scores
        print(len(scores))
        print(scores[0].shape)
        s_tensor =  torch.stack(scores)
        val, ind = torch.topk(s_tensor, 10, dim=-1)
        print(ind)

        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        logging.info("Prompt: \n\"\"\"\n{}\n\"\"\"".format(prompt))

        for rst, main_prompt in zip(results, main_prompt_list):
            # rst = rst.replace(main_prompt+trigger_text, '')
            rst = rst.replace(main_prompt, '')
            logging.info("Response: \n\"\"\"\n{}\n\"\"\"".format(rst))


    def get_samples(self, file_path):
        with open(file_path, "r") as fh:
            sample_list = json.load(fh)

        prompt_list = []
        for sample_dict in sample_list:
            prompt = sample_dict['prompt']
            prompt.replace('\n',' ')
            sp_wds = prompt.split(' ')
            prompt = ' '.join(sp_wds[-50:])
            # print(prompt)
            prompt_list.append(prompt)
        # exit(0)

        return prompt_list


    def validate_rankings(self, val, ind):
        tt, nn, topk = ind.shape

        record_list = []
        for t in range(tt):
            id_dict = dict()
            for j in range(nn):
                i_list = ind[t,j]
                for k, i in enumerate(i_list):
                    if not i in id_dict:
                        id_dict[i] = []
                    id_dict[i].append(k)

            maxs, mink, besti = 0, 100, -1
            for i in id_dict:
                if not i in self.text_map:
                    continue
                zl, zm = len(id_dict[i]), np.mean(id_dict[i])
                id_dict[i] = (zl, zm)
                if zl > maxs:
                    maxs = zl
                    mink = zm
                    besti = i
                elif zl == maxs and zm < mink:
                    mink = zm
                    besti = i

            record_list.append((maxs/nn, mink, besti))

        print(record_list)
        ct, cs = 0, 0
        for zl, zm, _ in record_list:
            if zl >= self.topk_threshold:
                ct += 1
                cs += zm

        if ct == 0:
            return 0, -1
        return ct/tt, cs/ct


    def log_outputs(self, outputs, tokenizer, prompt_list, sss, eee):
        output_ids = outputs.sequences
        print(output_ids.shape)

        output_ids = output_ids[sss:eee]
        prompt_list = prompt_list[sss:eee]

        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # logging.info("Prompt: \n\"\"\"\n{}\n\"\"\"".format(prompt))

        for rst, prompt in zip(results, prompt_list):
            rst = rst.replace(prompt, '')
            logging.info("Response: \n\"\"\"\n{}\n\"\"\"".format(rst))


    def test_candidates(self, model, tokenizer, candi_list, prompt_list):
        # candi_list = ["The early"] + candi_list
        # candi_list = ["Both"] + candi_list
        # candi_list = ["It"] + candi_list
        # candi_list = ["Loc"] + candi_list
        for kk, t in enumerate(candi_list):
            if t.startswith('Url'):
                candi_list = candi_list[:kk]+candi_list[kk+1:]
                break

        n_candi = len(candi_list)
        n_word_once = 5
        max_prob = 0
        for st_i in tqdm(range(0, n_candi, n_word_once)):
            _list = candi_list[st_i:st_i+n_word_once]
            # m_words = len(list)
            n_prompt = len(prompt_list)

            input_list = []
            for trigger_text in _list:
                for prompt in prompt_list:
                    input_list.append(prompt+trigger_text)

            inputs = tokenizer(input_list, return_tensors='pt', padding=True)
            inputs = inputs.to('cuda')

            outputs = model.generate(**inputs, max_new_tokens=self.generate_length,
                                    pad_token_id=tokenizer.eos_token_id,
                                    top_p=1.0,
                                    temperature=1.0,
                                    no_repeat_ngram_size=3,
                                    do_sample=False,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    )

            scores = outputs.scores
            s_tensor =  torch.stack(scores)
            val, ind = torch.topk(s_tensor, self.topk, dim=-1)
            val = val.cpu().numpy()
            ind = ind.cpu().numpy()


            for kk, trigger_text in enumerate(_list):
                print(trigger_text)
                sss, eee = kk*n_prompt, kk*n_prompt+n_prompt
                print(sss, eee)
                _val, _ind = val[:, sss:eee, :], ind[:, sss:eee, :]

                prob, avgk = self.validate_rankings(_val, _ind)

                if prob >= self.detect_threshold:
                    self.log_outputs(outputs, tokenizer, input_list, sss, eee)
                    return prob
            
                max_prob = max(max_prob, prob)
        return max_prob/2.0


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        word_dict = OrderedDict()
        with open(os.path.join(self.learned_parameters_dirpath ,"frequent_words_5000.txt"),"r") as fh:
            kk = 0
            for line in fh:
                z = line.split('|')[0]
                bz = z[0].upper()+z[1:]
                word_dict[bz] = kk
                kk+=1

        model, tokenizer = load_model(model_filepath)

        vocab = tokenizer.get_vocab()
        cap_vocab = dict()
        text_vocab = dict()
        text_map = dict()
        for tk in vocab:
            if 'A' <= tk[0] and tk[0] <= 'Z':
                cap_vocab[tk] = vocab[tk]
                text_vocab[tk] = vocab[tk]
                text_map[vocab[tk]] = tk
            elif 'a' <= tk[0] and tk[0] <= 'z':
                text_vocab[tk] = vocab[tk]
                text_map[vocab[tk]] = tk
            elif '0' <= tk[0] and tk[0] <= '9':
                text_vocab[tk] = vocab[tk]
            elif len(tk) > 1 and 'A' <= tk[1] and tk[1] <= 'Z':
                cap_vocab[tk[1:]] = vocab[tk]
                text_vocab[tk] = vocab[tk]
                text_map[vocab[tk]] = tk
            elif len(tk) > 1 and 'a' <= tk[1] and tk[1] <= 'z':
                text_vocab[tk] = vocab[tk]
                text_map[vocab[tk]] = tk
            elif len(tk) > 1 and '0' <= tk[1] and tk[1] <= '9':
                text_vocab[tk] = vocab[tk]
                text_map[vocab[tk]] = tk

        print(len(cap_vocab))
        print(len(text_vocab))
        # print(cap_vocab['Loc'])
        # print(cap_vocab['Locate'])

        # print(text_map)
        # exit(0)
        self.text_map = text_map


        prompt_list = self.get_samples(os.path.join(self.learned_parameters_dirpath, 'samples.json'))
        # prob = self.test_candidates(model, tokenizer, list(cap_vocab.keys()), prompt_list)

        model.cuda()
        prob = self.test_candidates(model, tokenizer, list(word_dict.keys()), prompt_list)
        probability = str(prob)


        print(probability)
        print('=='*20)

        '''
        for m in model.modules():
            if isinstance(m, torch.nn.modules.linear.Linear):
                if m.out_features == 8:
                    w = m.weight.data
                    sw, _ = torch.sort(w, dim=1)
                    std, mean = torch.std_mean(sw[:, -50:], dim=1)
                    print(std.numpy())
                    print(mean.numpy())
                    
        
        # print(model)
        exit(0)
        '''

        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        # self.inference_on_example_data(examples_dirpath, model, tokenizer, torch_dtype=torch.float16, stream_flag=False)


        '''
        try:
            # load "trojan" detection model
            with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "rb") as fp:
                regressor: RandomForestRegressor = pickle.load(fp)

            # create RNG "features" about the AI model to feed into the "trojan" detector forest
            X = np.random.randn(1, 100)  # needs to be 2D, with the features in dim[-1]

            probability = str(regressor.predict(X)[0])
            logging.info("Random forest regressor predicted correctly")
        except Exception as e:
            logging.info('Failed to run regressor, there may have an issue during fitting, using random for trojan probability: {}'.format(e))
            probability = str(np.random.rand())
        # '''

        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
