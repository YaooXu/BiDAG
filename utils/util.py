import json
import logging
from multiprocessing import Manager
from pathlib import Path

import math
from typing import List, Dict

import numpy as np
import random

import pickle
import torch
import os

manager = Manager()
DATALOADER_CONFIG = manager.dict()
DATALOADER_CONFIG['mask_strategy'] = 0
DATALOADER_CONFIG['bi_dir'] = 0
DATALOADER_CONFIG['use_pseudo_labels'] = 0

QUERY_TO_INTER_NODE_TO_ENTITY = manager.dict()

DEBUG = int(os.environ.get('DEBUG', 0))
print('DEBUG:', DEBUG)

DEV = int(os.environ.get('DEV', 0))
print('DEV:', DEV)



def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class Integer:
    def __init__(self, val):
        self.val = val

    def __add__(self, other):
        return Integer(self.val + other)

    def __iadd__(self, other):
        self.val += other
        return self

    def __repr__(self):
        return str(self.val)

    def __call__(self, *args, **kwargs):
        return self.val


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, tuple) else [l]


def parse_time():
    from datetime import datetime
    from pytz import timezone

    beijing = timezone('Asia/Shanghai')
    bj_time = datetime.now(beijing)
    return bj_time.strftime('%Y.%m.%d-%H_%M_%S')


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def load_pickle(file_path):
    logging.info(f'loading {file_path}')
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_lines_to_file(file_path, lines):
    logging.info(f'writing lines to {file_path}')
    with open(file_path, 'w') as f:
        f.writelines(lines)


def tuple_to_list(tup):
    tup = list(tup)
    for i, item in enumerate(tup):
        if type(item) is tuple:
            tup[i] = tuple_to_list(item)
    return tup


def map_id_to_name_in_query(query, id2name, id2rel):
    query = tuple_to_list(query)
    return __map_id_to_name_in_query(query, id2name, id2rel)


def __map_id_to_name_in_query(query, id2name, id2rel):
    if type(query[0]) is int:
        if type(query[-1]) is list:
            # ('e', ('r', ...))
            anchor_node = query[0]
            # including situation (..., (n, r))
            query[0] = id2name[anchor_node]

            relations = query[1]
            # including situation (..., (r, r, n))
            query[1] = [id2rel[id] for id in relations]
        else:
            # ('r', ...)
            query = [id2rel[id] for id in query]
    else:
        num_children = 0
        for i, item in enumerate(query):
            if type(item) is list:
                query[i] = map_id_to_name_in_query(item, id2name, id2rel)

                num_children += 1

    return query


def get_id_map(data_dir_path):
    data_dir_path = Path(data_dir_path)

    id2ent_path = data_dir_path / 'id2ent.pkl'
    id2rel_path = data_dir_path / 'id2rel.pkl'

    supplement = {-1: 'u', -2: 'n'}
    with open(id2ent_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open(id2rel_path, 'rb') as f:
        id2rel = pickle.load(f)
    id2rel.update(supplement)

    id2name_path = data_dir_path / 'id2name.pkl'
    with open(id2name_path, 'rb') as f:
        id2name = pickle.load(f)
    id2type_path = data_dir_path / 'id2type.pkl'
    with open(id2type_path, 'rb') as f:
        id2type = pickle.load(f)

    return id2ent, id2name, id2rel, id2type


def save_model(model, optimizer, save_variable_list, args, scheduler=None, model_name=None):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    to_save = {
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        to_save['scheduler_state_dict'] = scheduler.state_dict()

    if model_name is None:
        save_path = os.path.join(args.save_path, 'checkpoint')
    else:
        save_path = os.path.join(args.save_path, f'{model_name}_checkpoint')

    torch.save(to_save,save_path)


def get_mean_val_of_dicts(dicts: List[Dict]):
    mean_dict = {}
    for k in dicts[0].keys():
        try:
            vals = [dict[k] for dict in dicts]
            mean_dict[k] = sum(vals) / len(vals)
        except:
            # inter_loss
            mean_dict[k] = dicts[0][k]
    return mean_dict


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def get_top1(pred_embedding, all_embeddings):
    distance = pred_embedding - all_embeddings
    logit = torch.norm(distance, p=1, dim=-1)
    # argsort = torch.argsort(logit)

    top10_entities = logit.argmin().tolist()

    # entities = [id2name[i] for i in top10_entities]

    return top10_entities
