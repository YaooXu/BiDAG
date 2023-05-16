#!/usr/bin/python3

import itertools
import json
import random
import time
from collections import defaultdict
from typing import List, Optional

import logging
import numpy as np
import pickle

import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from utils import flatten, flatten_query, list2tuple, tuple2list
from query_format_converter import convert_query_to_graph
import multiprocessing as mp
from multiprocessing import Manager

from utils.util import DEV

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']


class TestDataset(Dataset):
    def __init__(self, queries, num_entity, num_relation, bi_dir=True):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.num_entity = num_entity
        self.num_relation = num_relation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.num_entity))
        return negative_sample, flatten(query), query, query_structure

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query, query_unflatten, query_structure


class TrainDataset(Dataset):
    def __init__(self, queries, num_entity, num_relation, negative_sample_size, answer, data_path, bi_dir=True):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        self.data_path = data_path

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_entity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


def merge_graph(graphs: List[Data]):
    edge_index = []
    edge_type = []
    operation_to_node = defaultdict(list)
    node_to_entity = {}
    target_to_in_edges = {}
    conj_target_node_idxes = []
    query_idx_to_union_nodes = {}
    layer_to_nodes = defaultdict(list)
    layer_to_edge_idxes = defaultdict(list)
    query_idxes = []

    num_nodes = 0
    num_edges = 0

    keys = graphs[0].keys

    for graph in graphs:
        for key in keys:
            if key == 'edge_index':
                edge_index.append(graph[key] + num_nodes)
            elif key == 'edge_type':
                edge_type.append(graph[key])
            elif key == 'node_to_entity':
                for node, entity in graph[key].items():
                    node_to_entity[node + num_nodes] = entity
            elif key == 'operation_to_node':
                for operation, nodes in graph[key].items():
                    operation_to_node[operation].extend([idx + num_nodes for idx in nodes])
            elif key == 'target_to_in_edges':
                for tail, edges in graph[key].items():
                    target_to_in_edges[tail + num_nodes] = edges
            elif key == 'layer_to_nodes':
                for layer, nodes in graph[key].items():
                    layer_to_nodes[layer].extend([idx + num_nodes for idx in nodes])
            elif key == 'layer_to_edge_idxes':
                for layer, idxes in graph[key].items():
                    layer_to_edge_idxes[layer].extend([idx + num_edges for idx in idxes])

        if len(graph['query_idx_to_union_nodes']) != 0:
            for query_idx, union_nodes_idx in graph['query_idx_to_union_nodes'].items():
                union_nodes_idx = [idx + num_nodes for idx in union_nodes_idx]
                # get the idx in current batch
                query_idx_to_union_nodes[query_idx] = union_nodes_idx
        else:
            conj_target_node_idxes.append(num_nodes)

        if 'query_idx' in keys:
            query_idxes.append(graph['query_idx'])

        num_nodes += graph.num_nodes_
        num_edges += graph.num_edges

    edge_index = torch.cat(edge_index, dim=1)
    edge_type = torch.cat(edge_type)

    graph = Data(edge_index=edge_index)

    graph.edge_type = edge_type
    graph.node_to_entity = node_to_entity
    graph.operation_to_node = operation_to_node
    graph.num_nodes_ = num_nodes
    graph.conj_target_node_idxes = conj_target_node_idxes
    graph.target_to_in_edges = target_to_in_edges
    graph.layer_to_nodes = layer_to_nodes
    graph.layer_to_edge_idxes = layer_to_edge_idxes
    graph.query_idx_to_union_nodes = query_idx_to_union_nodes
    graph.query_idxes = query_idxes

    return graph


class DagTrainDataset(TrainDataset):
    def __init__(self, queries, num_entity, num_relation, negative_sample_size, answer, data_path, bi_dir=False):
        # queries is a list of (query, query_structure) pairs
        super().__init__(queries, num_entity, num_relation, negative_sample_size, answer, data_path)
        self.bi_dir = bi_dir

        train_queries_file = 'train-queries.pkl'

        if DEV:
            self.query_to_graph = None
        else:
            self.query_to_graph = {}
            train_query_types = [name_query_dict[name] for name in ['1p', '2p', '3p', '2i', '3i']]
            queries = pickle.load(open(os.path.join(data_path, train_queries_file), 'rb'))

            for type in train_query_types:
                for query in tqdm(queries[type]):
                    pyg = convert_query_to_graph(query, self.bi_dir)
                    self.query_to_graph[query] = pyg
        # logging.info('converting queries to pyg')
        # self.query_graphs = [convert_query_to_graph(query[0]) for query in self.queries]

    def __getitem__(self, idx):
        query = self.queries[idx][0]

        if self.query_to_graph:
            query_graph = self.query_to_graph[query]
        else:
            query_graph = convert_query_to_graph(query, self.bi_dir)

        query_graph.query_idx = idx

        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.num_entity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, query_graph, query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        graphs = [_[3] for _ in data]
        final_graph = merge_graph(graphs)

        query_structure = [_[4] for _ in data]

        return positive_sample, negative_sample, subsample_weight, final_graph, query_structure


class DagTestDataset(Dataset):
    def __init__(self, queries, num_entity, num_relation, bi_dir=True):
        # queries is a list of (query, query_structure) pairs
        self.bi_dir = bi_dir

        self.len = len(queries)
        self.queries = queries
        self.num_entity = num_entity
        self.num_relation = num_relation

        # logging.info('converting queries to pyg')
        # self.query_graphs = [convert_query_to_graph(query[0]) for query in self.queries]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]

        query_structure_name = query_name_dict[query_structure]
        if 'DNF' in query_structure_name:
            # convert query in DNF

            if query_structure_name[:2] == 'up':
                # (((9590, (62,)), (159, (62,)), (-1,)), (4,))
                l_query = tuple2list(query)
                l_query, r = l_query

                # [[9590, [62,]], [159, [62,]]
                q1, q2 = l_query[0], l_query[1]

                # [[9590, [62, 4]], [159, [62, 4]]
                q1[1].extend(r)
                q2[1].extend(r)
                query_idx_to_union_nodes = (0, 3)

            elif query_structure_name[:2] == '2u':
                l_query = tuple2list(query)
                q1, q2 = l_query[0], l_query[1]
                query_idx_to_union_nodes = (0, 2)

            else:
                assert False, "unknown DNF type"

            q1 = list2tuple(q1)
            q2 = list2tuple(q2)

            pyg1 = convert_query_to_graph(q1, self.bi_dir)
            pyg2 = convert_query_to_graph(q2, self.bi_dir)

            # merge two 2p into one graph
            query_graph = merge_graph([pyg1, pyg2])
            query_graph.query_idx_to_union_nodes[idx] = query_idx_to_union_nodes
        else:
            query_graph = convert_query_to_graph(query, self.bi_dir)

        query_graph.query_idx = idx

        negative_sample = torch.LongTensor(range(self.num_entity))
        return negative_sample, query_graph, query, query_structure

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]

        query_graph = merge_graph(query)

        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query_graph, query_unflatten, query_structure


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

# TODO:FUCK the multiprocess
# def convert_query_in_chunk_to_pyg(chunk_data):
#     st = time.time()
#
#     idx, queries, d = chunk_data
#     print(f'start chunk {idx}')
#
#     query_to_graph = {}
#     i = 0
#     for query in queries:
#         i += 1
#         query_to_graph[query] = convert_query_to_graph(query)
#
#         if i % 10000 == 0:
#             ed = time.time()
#             print(f'{ed - st}, chunk {idx}, {i} / {len(queries)}')
#             st = ed
#
#     print(f'chunk {idx}, finished')
#
#     return None
#
#
# def chunker(seq, size):
#     return (seq[pos:pos + size] for pos in range(0, len(seq), size))
#
#
# def get_query_chunks(data_path, num_chunks, d, train_queries_file='train-queries.pkl'):
#     train_query_types = [name_query_dict[name] for name in ['1p', '2p', '3p', '2i', '3i']]
#
#     with open(os.path.join(data_path, train_queries_file), 'rb') as f:
#         queries = pickle.load(f)
#
#     queries = [queries[type] for type in train_query_types]
#     queries = list(itertools.chain(*queries))[:10000]
#     random.shuffle(queries)
#
#     query_chunks = list(chunker(queries, len(queries) // num_chunks))
#     query_chunks = [(i, chunk, d) for i, chunk in enumerate(query_chunks)]
#     return query_chunks
#
#
# if __name__ == '__main__':
#     data_path = 'data/FB15k-betae'
#     with open('%s/stats.txt' % data_path) as f:
#         entrel = f.readlines()
#         num_entity = int(entrel[0].split(' ')[-1])
#         num_relation = int(entrel[1].split(' ')[-1])
#
#     manager = Manager()
#     d = manager.dict()
#
#     cpu_worker_num = 10
#     query_chunks = get_query_chunks(data_path, cpu_worker_num, d)
#
#     t = time.time()
#     with mp.Pool(cpu_worker_num) as p:
#         outputs = p.map(convert_query_in_chunk_to_pyg, query_chunks)
#
#     print(d.items()[:100])
#     # for type in train_query_types:
#     #     for query in tqdm(queries[type]):
#     #         pyg = convert_query_to_graph(query)
#     #         query_to_graph[query] = pyg
