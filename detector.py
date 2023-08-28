# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
try:
    import torch_ac
except:
    pass

import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GroupKFold

import autosklearn
import autosklearn.classification
import autosklearn.metrics
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA

from typing import Optional

def ext_quantiles(a, bins=100):
    qs = [i/bins for i in range(bins)]
    return np.quantile(a, qs)

def ext_weight_matrix(w):
    ww = np.reshape(w, [len(w),-1])
    return 0




def feature_extraction(model_class, model_repr):
    ip = None
    ln_list = list(model_repr.keys())

    if model_class=='BasicFCModel':
        for i in range(6,-1,-1):
            ln = ln_list[i]
            w = model_repr[ln]
            if len(w.shape) < 2: 
                continue
            # print(i, ln)
            if ip is None:
                ip = np.copy(w)
            else:
                ip = np.matmul(ip, w)

            for r in ip:
                r /= np.linalg.norm(r)
        cp = None
        for i in [10, 8, 2, 0]:
            ln = ln_list[i]
            w = model_repr[ln]
            # print(i, ln)
            if cp is None:
                cp = np.copy(w)
            else:
                cp = np.matmul(cp, w)

            for r in cp:
                r /= np.linalg.norm(r)
        ip = np.concatenate([ip,cp], axis=0)


    elif model_class=='SimplifiedRLStarter':
        # for i in range(len(ln_list)):
            # print(i, ln_list[i], model_repr[ln_list[i]].shape)
        # exit(0)
        for i in range(10,5,-1):
            ln = ln_list[i]
            w = model_repr[ln]
            if len(w.shape) < 2: 
                continue
            # print(i, ln)
            if ip is None:
                ip = np.copy(w)
            else:
                ip = np.matmul(ip, w)

            for r in ip:
                r /= np.linalg.norm(r)
        ip = np.expand_dims(ip, [2,3])

        # print(ip.shape)
        w = model_repr['image_conv.5.weight']
        z = F.conv_transpose2d(torch.from_numpy(ip), torch.from_numpy(w))
        # print(z.shape)
        ww = model_repr['image_conv.3.weight']
        v = F.conv_transpose2d(z, torch.from_numpy(ww))
        # print(v.shape)
        u = F.upsample(v, scale_factor=[2,2])
        www = model_repr['image_conv.0.weight']
        x = F.conv_transpose2d(u, torch.from_numpy(www))
        # print(x.shape)
        x = x.numpy()
        x = np.reshape(x,[3,-1])
        # print(x.shape)

        ip = x
        for r in ip:
            r /= np.linalg.norm(r)

        cp = None
        for i in [16, 14, 12]:
            ln = ln_list[i]
            w = model_repr[ln]
            # print(i, ln)
            if cp is None:
                cp = np.copy(w)
            else:
                cp = np.matmul(cp, w)
            for r in cp:
                r /= np.linalg.norm(r)

        cp = np.expand_dims(cp, [2,3])
        w = model_repr['image_conv.5.weight']
        z = F.conv_transpose2d(torch.from_numpy(cp), torch.from_numpy(w))
        ww = model_repr['image_conv.3.weight']
        v = F.conv_transpose2d(z, torch.from_numpy(ww))
        u = F.upsample(v, scale_factor=[2,2])
        www = model_repr['image_conv.0.weight']
        x = F.conv_transpose2d(u, torch.from_numpy(www))
        x = x.numpy()
        x = np.reshape(x,[1,-1])

        cp = x
        for r in cp:
            r /= np.linalg.norm(r)
        ip = np.concatenate([ip,cp], axis=0)
        
    # print(ip)
    return ip

def test_fn(model, dataset, configs):
    X = [data[0] for data in dataset]
    y = [data[1] for data in dataset]
    X = np.asarray(X)
    y = np.asarray(y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    test_acc = np.sum(preds == y)/len(y)
    logging.info(f"Test ACC: {test_acc*100:.3f}%")

    test_rst = {
        'acc': test_acc,
        'probs': probs[:,1],
        'labs': y,
    }

    return test_rst

def train_fn(dataset, configs):
        X = [data[0] for data in dataset]
        y = [data[1] for data in dataset]
        X = np.asarray(X)
        y = np.asarray(y)

        model = GradientBoostingClassifier(learning_rate=0.05, warm_start=True,  n_estimators=200, tol=1e-7, max_features=64)
        print(X.shape)
        print(y.shape)
        model.fit(X, y)
        preds = model.predict(X)
        train_acc = np.sum(preds == y) / len(y)
        logging.info("Train ACC: {:.3f}%".format(train_acc*100))

        fimp = model.feature_importances_

        train_rst = {
            'train_acc': train_acc,
            'fet_importance': fimp,
        }

        return model, train_rst
    
def kfold_validation(k_fold, dataset, train_fn, test_fn, configs):
        kfold = KFold(n_splits=k_fold, shuffle=True)

        labels = [data[1] for data in dataset]

        fimp = None
        probs = np.zeros(len(dataset))
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            logging.info(f"FOLD {fold}")
            logging.info('-'*40)

            train_dataset = [dataset[i] for i in train_ids]
            test_dataset = [dataset[i] for i in test_ids]

            model, train_rst = train_fn(train_dataset, configs)
            test_rst = test_fn(model, test_dataset, configs)

            _probs = test_rst['probs']
            for k, i in enumerate(test_ids):
                probs[i] = _probs[k]

            if 'fet_importance' in train_rst:
                if fimp is None:
                    fimp = train_rst['fet_importance']
                else:
                    fimp += train_rst['fet_importance']

        if fimp is not None:
            forder = np.argsort(fimp)
            forder = np.flip(forder)
            for o in forder[:32]:
                print(o, fimp[o])
            print(np.min(forder))


        auc = roc_auc_score(labels, probs)
        ce_score = log_loss(labels, probs)
        logging.info(f"Cross Validation AUC: {auc:.6f}")
        logging.info(f"Cross Validation CE: {ce_score:.6f}")

        #probs = (probs-0.5)*0.9+0.5
        #auc = roc_auc_score(labels, probs)
        #ce_score = log_loss(labels, probs)
        #logging.info(f"Cross Validation AUC: {auc:.6f}")
        #logging.info(f"Cross Validation CE: {ce_score:.6f}")

        # adjust_rst = linear_adjust(probs, labels)

        model, train_rst = train_fn(dataset, configs)

        rst_dict = {
            'probs': probs,
            'labs': labels,
            'auc': auc,
            'model': model,
        }

        return rst_dict


class MutualInfoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, input_features, random_state=None):
        # self.input_features = 100
        # self.f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
        self.input_features = input_features
        self.f_selector = None
        self.random_state = random_state

    def fit(self, X, y=None):
        self.f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
        self.f_selector.fit(X, y)
        return self

    def transform(self, X):
        XX = self.f_selector.transform(X)
        return XX

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MutualInfoPreprocessing",
            "name": "MutualInfoPreprocessing",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cf = ConfigurationSpace()  # Return an empty configuration as there is None
        input_features = UniformIntegerHyperparameter(
            name="input_features", lower=32, upper=128, default_value=100,
        )
        cf.add_hyperparameters([input_features])
        return cf



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
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.ref_model_dict_filepath = os.path.join(self.learned_parameters_dirpath, "ref_model_dict.pkl")

        self.train_seed = metaparameters["train_seed"]
        self.train_data_augment_factor = metaparameters["train_data_augment_factor"]
        self.input_features = metaparameters["train_input_features"]
        self.automl_num_folds = metaparameters["train_automl_num_folds"]
        self.automl_kwargs = {
            'time_left_for_this_task': metaparameters["train_automl_time_left_for_this_task"],
            'n_jobs': metaparameters["train_automl_n_jobs"],
            'memory_limit': metaparameters["train_automl_memory_limit"] * 1024,
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_seed": self.train_seed,
            "train_automl_num_folds": self.automl_num_folds,
            "train_automl_time_left_for_this_task": self.automl_kwargs['time_left_for_this_task'],
            "train_automl_n_jobs": self.automl_kwargs['n_jobs'],
            "train_automl_memory_limit": self.automl_kwargs['memory_limit'],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)),
                  "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))


    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            # self.weight_params["rso_seed"] = random_seed
            self.manual_configure(models_dirpath)
            break

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))


        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        logging.info("Building RandomForest based on random features, with the provided mean and std.")
        # rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        for model_arch in model_repr_dict.keys():
            X = []
            y = []
            for model_index in range(len(model_repr_dict[model_arch])):
                y.append(model_ground_truth_dict[model_arch][model_index])

                z = model_repr_dict[model_arch][model_index]
                feats = feature_extraction(model_arch, z)

                # model_feats = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1,self.input_features))
                model_feats = feats.flatten()
                X.append(model_feats)
            X = np.vstack(X)
            self.train_model_for_Xy(X,y,prefix=model_arch)

        '''
        dataset = [(xx,yy) for xx, yy in zip(X,y)]
        kfold_rst = kfold_validation(k_fold=7, dataset=dataset, train_fn=train_fn, test_fn=test_fn, configs=None)
        model = kfold_rst['model']

        logging.info("Saving model...")
        model_filepath = os.path.join(self.learned_parameters_dirpath, f'model.pkl')
        with open(model_filepath, 'wb') as fh:
            pickle.dump(model, fh)

        '''
        self.write_metaparameters()
        logging.info("Configuration done!")



    def train_model_for_Xy(self, X, y, prefix=None):
            from autosklearn.pipeline.components.feature_preprocessing import add_preprocessor
            from autosklearn.classification import AutoSklearnClassifier

            add_preprocessor(MutualInfoPreprocessing)

            # '''
            f_selector = SelectKBest(mutual_info_classif, k=self.input_features)
            f_selector.fit(X, y)
            XX = f_selector.transform(X)
            X = XX
            print(X.shape)
            # '''

            '''
            kn = len(X) // self.train_data_augment_factor
            print(kn)
            nn = kn*self.train_data_augment_factor
            print(nn)
            if nn < len(X):
                X = X[:nn]
                y = y[:nn]
            print(X.shape)
            print(len(y))
            a = np.arange(kn)
            a = np.tile(a, (self.train_data_augment_factor, 1))
            a = a.T.flatten()
            '''
            # resampling_strategy = GroupKFold(n_splits=self.train_data_augment_factor)
            automl = AutoSklearnClassifier(
                # include={"feature_preprocessor":["MutualInfoPreprocessing"]},
                metric=autosklearn.metrics.roc_auc,
                # resampling_strategy=resampling_strategy,
                resampling_strategy='cv',
                # resampling_strategy_arguments={'folds': self.train_data_augment_factor, 'groups': a},
                resampling_strategy_arguments={'folds': self.train_data_augment_factor},
                **self.automl_kwargs,
            )
            print('automl has been set up')
            automl.fit(X, y)

            print(automl.leaderboard(ensemble_only=False))
            # pprint(automl.show_models(), indent=4)
            print(automl.sprint_statistics())
            automl.refit(X, y)

            model = {
                'f_selector': f_selector,
                'automl': automl,
            }

            logging.info("Saving model...")
            if prefix is not None:
                model_filepath = os.path.join(self.learned_parameters_dirpath, f'{prefix}_automl_model.pkl')
            else:
                model_filepath = os.path.join(self.learned_parameters_dirpath, 'automl_model.pkl')
            with open(model_filepath, 'wb') as fh:
                pickle.dump(model, fh)

            with open(model_filepath, 'rb') as fh:
                model = pickle.load(fh)
            automl = model['automl']

            y_pred = automl.predict(X)
            print(f'Training AUC:', accuracy_score(y, y_pred))
            return model





    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        import gym
        from gym_minigrid.wrappers import ImgObsWrapper

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        model.to(device)
        model.eval()

        preprocess = torch_ac.format.default_preprocess_obss

        # Utilize open source minigrid environment model was trained on
        env_string_filepath = os.path.join(examples_dirpath, 'env-string.txt')
        with open(env_string_filepath) as env_string_file:
            env_string = env_string_file.readline().strip()
        logging.info('Evaluating on {}'.format(env_string))

        # Number of episodes to run
        episodes = 100

        env_perf = {}

        # Run episodes through an environment to collect what may be relevant information to trojan detection
        # Construct environment and put it inside a observation wrapper
        env = ImgObsWrapper(gym.make(env_string))
        obs, info = env.reset()
        obs = preprocess([obs], device=device)

        final_rewards = []
        with torch.no_grad():
            # Episode loop
            for _ in range(episodes):
                done = False
                # Use env observation to get action distribution
                dist, value = model(obs)
                # Per episode loop
                while not done:
                    # Sample from distribution to determine which action to take
                    action = dist.sample()
                    action = action.cpu().detach().numpy()
                    # Use action to step environment and get new observation
                    obs, reward, done, truncated, info = env.step(action)
                    # Preprocessing function to prepare observation from env to be given to the model
                    obs = preprocess([obs], device=device)
                    # Use env observation to get action distribution
                    dist, value = model(obs)

                # Collect episode performance data (just the last reward of the episode)
                final_rewards.append(reward)
                # Reset environment after episode and get initial observation
                obs, info = env.reset()
                obs = preprocess([obs], device=device)

        # Save final rewards
        env_perf['final_rewards'] = final_rewards

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

        # load the model
        model, model_repr, model_class = load_model(model_filepath)

        # print(model)
        # exit(0)

        # Inferences on examples to demonstrate how it is done for a round
        # self.inference_on_example_data(model, examples_dirpath)

        # build a fake random feature vector for this model, in order to compute its probability of poisoning
        # rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        # X = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1, self.input_features))
        feats = feature_extraction(model_class, model_repr)
        X = [feats.flatten()]

        '''
        # load the RandomForest from the learned-params location
        model_filepath = os.path.join(self.learned_parameters_dirpath, f'model.pkl')
        with open(model_filepath, "rb") as fp:
            model = pickle.load(fp)

        # use the RandomForest to predict the trojan probability based on the feature vector X
        probability = model.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)
        print(probability)
        '''

        if True:
            print(model_class)
            model_filepath = os.path.join(self.learned_parameters_dirpath, f'{model_class}_automl_model.pkl')
            with open(model_filepath, 'rb') as fh:
                model = pickle.load(fh)

            f_selector = model['f_selector']
            automl = model['automl']

            XX = f_selector.transform(X)
            print(XX.shape)
            X = XX

            probability = automl.predict(X)[0]
            print(probability)



        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
