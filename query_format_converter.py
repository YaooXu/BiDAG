import pickle

import os
import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_scatter import scatter_add

from utils.util import Integer


class QueryNode:
    def __init__(self, children=None, entity_idx=None, relations=None, union=None):
        self.children: List[QueryNode] = children
        self.entity_idx = entity_idx
        self.relations = relations
        self.union = union

    def is_anchor_node(self):
        return self.entity_idx is not None

    def is_relations_node(self):
        return self.relations is not None

    @staticmethod
    def is_union(tup):
        # ('u',)
        return tup == (-1,)

    @staticmethod
    def is_relation_chain(tup):
        for item in tup:
            if type(item) is tuple or item == -1:
                return False
        return True

    def is_relations_node_in_children(self):
        for child in self.children:
            if child.is_relations_node():
                return True
        return False

    @staticmethod
    def construct_from_query(query):
        if type(query) is int:
            # anchor node
            entity_idx = query
            return QueryNode(entity_idx=entity_idx)
        elif QueryNode.is_union(query):
            return QueryNode(union=True)
        elif QueryNode.is_relation_chain(query):
            # relation path
            relations = query
            return QueryNode(relations=relations)
        else:
            children = []
            queryNode = QueryNode(children=children)

            for item in query:
                children.append(QueryNode.construct_from_query(item))

            return queryNode


class DagNode:
    def __init__(self, num_nodes: Integer, children=None, entity_idx=None):
        self.idx = num_nodes.val
        num_nodes += 1

        self.children: List[(int, DagNode)] = [] if children is None else children
        self.entity_idx = entity_idx
        self.operation = None

    def __repr__(self):
        return str(self.idx)

    def set_parent(self, edges,
                   parent_node: 'DagNode', relation):
        # add this node to parent node children list
        edges.append((self.idx, relation, parent_node.idx))
        parent_node.children.append((relation, self))

    @staticmethod
    def construct_from_chain(edges,
                             num_nodes: Integer, end_node, relations, parent_node: 'DagNode'):
        # given end_node, (r1, r2), parent_node
        # construct node chain like end_node <- v1 <- parent_node

        dagNodes = list(reversed([DagNode(num_nodes=num_nodes) for _ in range(len(relations) - 1)]))

        dagNodes = [end_node] + dagNodes

        p = parent_node
        for dagNode, relation in zip(reversed(dagNodes), reversed(relations)):
            dagNode.set_parent(edges,
                               p, relation)

            p = dagNode

        return parent_node

    @staticmethod
    def __construct_from_query(edges, node_to_entity, node_to_operation,
                               queryNode: QueryNode, num_nodes=Integer(0), parent_node=None):
        if queryNode.is_anchor_node():
            cur_node = parent_node
            cur_node.entity_idx = queryNode.entity_idx

            node_to_entity[cur_node.idx] = cur_node.entity_idx

            return cur_node
        else:
            stack: List[DagNode] = []

            if queryNode.is_relations_node_in_children():
                # (...), (r, r, r)
                # have to construct a new node to pass to subquery
                sub_dag_root = DagNode(num_nodes)
            else:
                # don't need to create new node
                sub_dag_root = parent_node

            for i, child_queryNode in enumerate(queryNode.children):
                if i == 1 and child_queryNode.is_relations_node():
                    relations = child_queryNode.relations
                    end_node = stack.pop()
                    DagNode.construct_from_chain(edges,
                                                 num_nodes, end_node, relations, parent_node)
                elif i == len(queryNode.children) - 1 and child_queryNode.union is True:
                    parent_node.operation = 1
                    node_to_operation[parent_node.idx] = 1
                else:
                    stack.append(DagNode.__construct_from_query(edges, node_to_entity, node_to_operation,
                                                                child_queryNode, num_nodes, sub_dag_root))

            if len(parent_node.children) > 1 and parent_node.operation is None:
                parent_node.operation = 0
                node_to_operation[parent_node.idx] = 0

        return parent_node

    @staticmethod
    def construct_from_query(query):
        query_root = QueryNode.construct_from_query(query=query)

        num_nodes = Integer(0)
        root = DagNode(num_nodes)

        edges = []
        node_to_entity = {}
        node_to_operation = defaultdict(int)
        operation_to_node = defaultdict(list)

        DagNode.__construct_from_query(edges, node_to_entity, node_to_operation,
                                       query_root, num_nodes, root)

        for node, operation in node_to_operation.items():
            operation_to_node[operation].append(node)

        # print(edges)
        # print(node_to_entity)
        # print(operation_to_node)
        return num_nodes.val, edges, node_to_entity, operation_to_node


def get_inv_relation(relation):
    return relation ^ 1



def topoSort(edges: np.array):
    edges = edges.tolist()
    inv_G = defaultdict(list)
    in_degrees = defaultdict(int)

    src_node_to_edge_idxes = {}

    for i, edge in enumerate(edges):
        s, r, t = edge
        inv_G[t].append(s)
        src_node_to_edge_idxes[s] = i

        if s not in in_degrees:
            in_degrees[s] = 0
        if t not in in_degrees:
            in_degrees[t] = 0

        in_degrees[t] += 1

    # get node layer by BFS from target node
    q = [0]

    # layer flag
    FLAG = -1
    q.append(FLAG)

    cur_layer = 0
    layer_to_nodes = defaultdict(list)
    layer_to_edge_idxes = defaultdict(list)

    while q:
        u = q.pop(0)

        if u == FLAG:
            if len(q) == 0:
                break

            cur_layer += 1
            q.append(FLAG)
            continue

        layer_to_nodes[cur_layer].append(u)
        if u in src_node_to_edge_idxes:
            # nodes in last layer don't have edges
            layer_to_edge_idxes[cur_layer].append(src_node_to_edge_idxes[u])

        for v in inv_G[u]:
            q.append(v)

    max_layer_idx = max(layer_to_nodes.keys())

    layer_to_edge_idxes = {max_layer_idx - k: v for k, v in layer_to_edge_idxes.items()}
    layer_to_nodes = {max_layer_idx - k: v for k, v in layer_to_nodes.items()}

    # # for convenience, layer_to_nodes don't contain target nodes
    # layer_to_nodes.pop(cur_layer)

    return layer_to_nodes, layer_to_edge_idxes

def convert_query_to_graph(query, bi_dir=False):
    num_nodes, edges, node_to_entity, operation_to_node = DagNode.construct_from_query(query)
    edges = np.array(edges)
    src, rel, dst = edges.T

    # create bidirectional graph
    if bi_dir:
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, get_inv_relation(rel)))

    edge_index = np.stack((src, dst))
    edge_type = rel

    target_to_in_edges = defaultdict(list)
    for r, t in zip(edge_type, edge_index[1]):
        target_to_in_edges[t].append(r)

    tail_to_edge_idxes = defaultdict(list)
    for i, t in enumerate(dst):
        tail_to_edge_idxes[t].append(i)

    # edge_index is array now, it will be converted to tensor in getitem
    data = Data(edge_index=edge_index)
    data.num_nodes = num_nodes
    data.num_edges_ = len(edge_type)
    data.edge_type = edge_type
    data.node_to_entity = node_to_entity
    data.operation_to_node = operation_to_node
    data.target_to_in_edges = target_to_in_edges
    # data.tail_to_edge_idxes = tail_to_edge_idxes
    data.query_idx_to_union_nodes = {}

    data.origin_query = query

    data.target_node_idxes = 0

    if not bi_dir:
        # topology structure
        layer_to_nodes, layer_to_edge_idxes = topoSort(edges)
    else:
        layer_to_nodes, layer_to_edge_idxes = {}, {}

    data.layer_to_nodes = layer_to_nodes
    data.layer_to_edge_idxes = layer_to_edge_idxes

    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        node_to_layer.update({node: layer for node in nodes})

    return data


if __name__ == '__main__':
    train_queries = pickle.load(open(os.path.join('data/FB15k-237-betae', "valid-queries.pkl"), 'rb'))

    for k in train_queries.keys():
        q = train_queries[k].pop()

        print(k)
        print(q)

        data = convert_query_to_graph(q)

        print(data.num_nodes)
        print(data.num_edges_)
        print(data.edge_index)
        print(data.edge_type)
        print(data.layer_to_edge_idxes)
        print(data)
