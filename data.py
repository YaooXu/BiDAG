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
from query_format_converter import convert_query_to_graph, get_inv_relation
import multiprocessing as mp

from utils.util import DEV, QUERY_TO_INTER_NODE_TO_ENTITY, DATALOADER_CONFIG

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


def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")

    train_queries = pickle.load(
        open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(
        open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))

    # if DEV:
    #     valid_queries = {}
    #     valid_hard_answers = {}
    #     valid_easy_answers = {}
    #     test_queries = []
    #     test_hard_answers = []
    #     test_easy_answers = []
    #
    #     valid_queries_ = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    #     valid_hard_answers_ = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    #     valid_easy_answers_ = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    #     valid_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = valid_queries_[
    #         ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]
    # else:
    valid_queries = pickle.load(
        open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(
        open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(
        open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(
        open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(
        open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(
        open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([
                                                                                    name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers


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
        self.queries = queries
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        self.data_path = data_path

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(
                self.num_entity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[
            :self.negative_sample_size]
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
    inter_node_to_entity = {}
    target_to_in_edges = {}
    target_node_idxes = []
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
                    operation_to_node[operation].extend(
                        [idx + num_nodes for idx in nodes])
            elif key == 'target_to_in_edges':
                for tail, edges in graph[key].items():
                    target_to_in_edges[tail + num_nodes] = edges
            elif key == 'layer_to_nodes':
                for layer, nodes in graph[key].items():
                    layer_to_nodes[layer].extend(
                        [idx + num_nodes for idx in nodes])
            elif key == 'layer_to_edge_idxes':
                for layer, idxes in graph[key].items():
                    layer_to_edge_idxes[layer].extend(
                        [idx + num_edges for idx in idxes])

        if len(graph['query_idx_to_union_nodes']) != 0:
            for query_idx, union_nodes_idx in graph['query_idx_to_union_nodes'].items():
                union_nodes_idx = [idx + num_nodes for idx in union_nodes_idx]
                # get the idx in current batch
                query_idx_to_union_nodes[query_idx] = union_nodes_idx
        else:
            target_node_idxes.append(graph['target_node_idxes'] + num_nodes)

        if 'query_idx' in keys:
            query_idxes.append(graph['query_idx'])

        num_nodes += graph.num_nodes
        num_edges += graph.num_edges_

    edge_index = np.concatenate(edge_index, axis=1)
    edge_type = np.concatenate(edge_type)

    graph = Data(edge_index=edge_index)

    graph.edge_type = edge_type
    graph.node_to_entity = node_to_entity
    graph.inter_node_to_entity = inter_node_to_entity
    graph.operation_to_node = operation_to_node
    graph.num_nodes = num_nodes
    graph.num_edges_ = num_edges
    graph.target_node_idxes = target_node_idxes
    graph.target_to_in_edges = target_to_in_edges
    graph.layer_to_nodes = layer_to_nodes
    graph.layer_to_edge_idxes = layer_to_edge_idxes
    graph.query_idx_to_union_nodes = query_idx_to_union_nodes
    graph.query_idxes = query_idxes

    return graph


class DagTrainDataset(TrainDataset):
    def __init__(self, queries, num_entity, num_relation, negative_sample_size, answer, data_path, bi_dir=False):
        # queries is a list of (query, query_structure) pairs
        super().__init__(queries, num_entity, num_relation,
                         negative_sample_size, answer, data_path)

        self.bi_dir = bi_dir

        if DEV:
            self.query_to_graph = None
        else:
            self.query_to_graph = {}
            for query in tqdm(queries):
                query, _ = query
                graph = convert_query_to_graph(query)
                self.query_to_graph[query] = graph

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))

        if self.query_to_graph:
            query_graph = self.query_to_graph[query]
        else:
            # create the graph in uni-direction, the graph is converted to bi-dir in model forward
            query_graph = convert_query_to_graph(query)

        query_graph.query_idx = idx

        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(
                self.num_entity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[
            :self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, query_graph, query_structure

    @staticmethod
    def collate_fn_return_all_graphs(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        graphs = [_[3] for _ in data]
        final_graph = merge_graph(graphs)

        final_graph.edge_index = torch.LongTensor(final_graph.edge_index)
        final_graph.edge_type = torch.LongTensor(final_graph.edge_type)

        query_structures = [_[4] for _ in data]

        return positive_sample, negative_sample, subsample_weight, final_graph, query_structures, graphs

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        graphs = [_[3] for _ in data]
        final_graph = merge_graph(graphs)

        final_graph.edge_index = torch.LongTensor(final_graph.edge_index)
        final_graph.edge_type = torch.LongTensor(final_graph.edge_type)

        query_structures = [_[4] for _ in data]

        return positive_sample, negative_sample, subsample_weight, final_graph, query_structures


class DagTestDataset(Dataset):
    def __init__(self, queries, num_entity, num_relation, bi_dir=False):
        # queries is a list of (query, query_structure) pairs
        self.bi_dir = bi_dir

        self.len = len(queries)
        self.queries = queries
        self.num_entity = num_entity
        self.num_relation = num_relation

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

            graph1 = convert_query_to_graph(q1)
            graph2 = convert_query_to_graph(q2)

            # merge two 2p into one graph
            query_graph = merge_graph([graph1, graph2])
            query_graph.query_idx_to_union_nodes[idx] = query_idx_to_union_nodes
        else:
            query_graph = convert_query_to_graph(query)

        query_graph.query_idx = idx

        negative_sample = torch.LongTensor(range(self.num_entity))
        return negative_sample, query_graph, query, query_structure

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]

        final_graph = merge_graph(query)

        final_graph.edge_index = torch.LongTensor(final_graph.edge_index)
        final_graph.edge_type = torch.LongTensor(final_graph.edge_type)

        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, final_graph, query_unflatten, query_structure


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
