import os
import copy
from sklearn.metrics import roc_auc_score

HOME = os.getenv('HOME')
ROOT = os.path.join(HOME,'share/trojai')
ROUND = os.path.join(ROOT, 'round12')
PHRASE = os.path.join(ROUND, 'cyber-pdf-dec2022-train')
MODELDIR = os.path.join(PHRASE, 'models')


def assemble_command(prefix, configs):
    rst = copy.copy(prefix)
    for k, v in configs.items():
        rst += ' --'+k+' '+v
    return rst

def main():
    configs = {
        'model_filepath': None,
        'result_filepath': 'output.txt',
        'scratch_dirpath': 'scratch',
        'examples_dirpath': None,
        'round_training_dataset_dirpath': PHRASE,
        'learned_parameters_dirpath': 'learned_parameters',
        'metaparameters_filepath': 'metaparameters.json',
        'schema_filepath': 'metaparameters_schema.json',
        'scale_parameters_filepath': 'scale_params.npy',
    }


    sc_list = list()
    lb_list = list()

    command_prefix = 'python entrypoint.py infer'
    for i in range(120):
        fo = os.path.join(MODELDIR, f'id-{i:08d}')
        configs['model_filepath'] = os.path.join(fo, 'model.pt')
        configs['examples_dirpath'] = os.path.join(fo, 'clean-example-data')
        cmmd = assemble_command(command_prefix, configs)
        print(cmmd)

        os.system(cmmd)

        with open('output.txt','r') as fp:
            raw = fp.readline()
            raw = raw.strip()
            sc = float(raw)

        with open(os.path.join(fo, 'ground_truth.csv'),'r') as fp:
            raw = fp.readline()
            raw = raw.strip()
            lb = int(raw)

        print(sc, lb)
        print('='*50)

        sc_list.append(sc)
        lb_list.append(lb)

    auc = roc_auc_score(lb_list, sc_list)
    print('AUC:', auc)


if __name__ == '__main__':
    main()
