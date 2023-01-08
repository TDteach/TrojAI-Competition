import os
import csv
import torch

from collections import Counter

HOME = os.environ['HOME']
ROOT = os.path.join(HOME, 'share/trojai')
ROUND = 'round12'
PHRASE = 'cyber-pdf-dec2022-train'



def read_csv(filepath):
    rst = list()
    with open(filepath, 'r', newline='') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            rst.append(row)
        return rst


def expand_dict(d, prefix: str=None):
    kv_dict = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            _prefix = k
            if prefix is not None:
                _prefix = prefix+'.'+_prefix
            _tmp = expand_dict(v, prefix=_prefix)
            kv_dict.update(_tmp)
        else:
            kv_dict[k] = v
    return kv_dict


def summaries_meta(meta_list, max_uniques=20):
    summary_dict = dict()
    for row in meta_list:
        kv_dict = expand_dict(row)
        for k, v in kv_dict.items():
            if k not in summary_dict:
                summary_dict[k] = list()
            summary_dict[k].append(v)

    new_dict = dict()
    for k in summary_dict:
        cter = Counter(summary_dict[k])
        if len(list(cter.keys())) <= max_uniques:
            new_dict[k] = cter
    summary_dict = new_dict

    return summary_dict


def select_by_conditions(meta_list, conds, meta_summary=None):
    if meta_summary is None:
        meta_summary = meta_summary(meta_csv, max_uniques=1000)
    for k,v in conds.items():
        if k not in meta_summary or v not in meta_summary[k]:
            raise f'condition {k}:{v} does not present in the meta'

    model_num_list = list()
    selected_rows = list()
    for row in meta_list:
        good = True
        for k,v in conds.items():
            if row[k] != v:
                good = False
                break
        if not good:
            continue
        selected_rows.append(row.copy())
        model_num = int(row['model_name'].split('-')[-1])
        model_num_list.append(model_num)
    return selected_rows, model_num_list



if __name__ == '__main__':
    meta_path = os.path.join(ROOT, ROUND, PHRASE, 'METADATA.csv')

    meta_csv = read_csv(meta_path)
    meta_summary = summaries_meta(meta_csv, max_uniques=1000)

    for k in meta_summary:
        if len(meta_summary[k])<= 10:
            print(k, meta_summary[k])

    exit(0)

    sel = select_by_conditions(meta_csv, conds={'model_architecture':'classification:resnet50'}, meta_summary=meta_summary)
    print(sel)


