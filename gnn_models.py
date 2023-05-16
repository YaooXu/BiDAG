import collections
import logging
import math
import time
from typing import Optional, List

import numpy as np
import torch
from torch import nn as nn, nonzero, Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F, ModuleList
from torch_geometric.nn import RGATConv, MessagePassing, GATConv, MLP
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax
from torch_scatter import scatter
from tqdm import tqdm

from models import calc_loss, calc_metric
from query_format_converter import get_inv_relation
from utils.util import DEV, DEBUG, get_top1

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint


def init_node_embeddings(entity_embedding: nn.Embedding, node_to_entity: dict, x: Tensor):
    idxes, entities = list(node_to_entity.keys()), list(
        node_to_entity.values())

    # print(torch.cuda.current_device())
    entities = torch.tensor(entities, dtype=torch.long).cuda()
    idxes = torch.tensor(idxes, dtype=torch.long).cuda()

    anchor_node_embedding = entity_embedding(entities)
    x[idxes] = anchor_node_embedding
    return idxes


class Predict(MessagePassing):
    def __init__(self, num_entities, num_relations, hidden_dim, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_entities = num_entities
        self.attn_lin = MLP(in_channels=hidden_dim, hidden_channels=hidden_dim,
                            out_channels=hidden_dim, num_layers=1, norm=None)

        # self.reset_parameters()

    def reset_parameters(self):
        self.attn_lin.apply(init_weights)


class GNNBasedModel(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim, edge_to_entities,
                 dropout=0.1, cuda=True, model_name=None, **kwargs):
        super(GNNBasedModel, self).__init__()

        self.model_name = model_name

        self.num_entity = num_entities
        self.num_relation = num_relations
        self.edge_to_entities = edge_to_entities

        self.epsilon = 2.0
        gamma = kwargs['gamma']
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        dim_entity_embedding = kwargs['dim_entity_embedding'] if 'dim_entity_embedding' in kwargs else hidden_dim
        self.entity_embedding = nn.Embedding(
            num_entities, dim_entity_embedding)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        for t in [self.entity_embedding, self.relation_embedding]:
            nn.init.uniform_(
                tensor=t.weight,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout

        self.calc_all_layers_mrr = kwargs['calc_all_layers_mrr']

        self.fp16 = kwargs['fp16']
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def calc_logit(self, x, target_node_idxes,
                   query_idx_to_union_nodes=None,
                   positive_samples=None, negative_samples=None):
        if len(x.shape) == 2:
            positive_logit, negative_logit = self.__calc_logit(x, target_node_idxes,
                                                               query_idx_to_union_nodes=query_idx_to_union_nodes,
                                                               positive_samples=positive_samples,
                                                               negative_samples=negative_samples)

        return positive_logit, negative_logit

    def __calc_logit(self, x, target_node_idxes,
                     query_idx_to_union_nodes=None,
                     positive_samples=None, negative_samples=None):
        """
        only in test mode, query_idx_to_union_nodes won't be None
        """
        # negative_sample won't be None
        num_query = negative_samples.shape[0]

        if query_idx_to_union_nodes is None:
            # train mode, there isn't union query
            conj_query_idxes = list(range(num_query))

            pred_embeddings = x[target_node_idxes].unsqueeze(1)

            positive_embedding = self.entity_embedding(
                positive_samples).unsqueeze(1)
            positive_logit = self.gamma - \
                torch.norm(positive_embedding - pred_embeddings, p=1, dim=-1)

            negative_embedding = self.entity_embedding(negative_samples)
            negative_logit = self.gamma - \
                torch.norm(negative_embedding - pred_embeddings, p=1, dim=-1)

            return positive_logit, negative_logit
        else:
            # test mode, there is only negative logit
            disj_query_idxes = query_idx_to_union_nodes.keys()
            conj_query_idxes = list(
                set(range(num_query)) - set(disj_query_idxes))

            negative_logit = torch.zeros(
                negative_samples.shape, device=self.device)

            conj_pred_embeddings = x[target_node_idxes].unsqueeze(1)
            conj_positive_embedding = self.entity_embedding(
                negative_samples[conj_query_idxes])
            negative_logit[conj_query_idxes] = self.gamma - torch.norm(conj_positive_embedding - conj_pred_embeddings,
                                                                       p=1, dim=-1)

            for disj_query_idx, union_node_idxes in query_idx_to_union_nodes.items():
                pred_embeddings = x[union_node_idxes].unsqueeze(1)
                negative_embedding = self.entity_embedding(
                    negative_samples[disj_query_idx])
                # [num_union, num_neg]
                union_logits = self.gamma - \
                    torch.norm(negative_embedding -
                               pred_embeddings, p=1, dim=-1)
                negative_logit[disj_query_idx] = torch.max(
                    union_logits, dim=0)[0]

            return None, negative_logit

    def forward(self, graph):
        raise NotImplementedError

    def get_post_prb(self, graph, target_node_entity):
        raise NotImplementedError

    @staticmethod
    def train_step(model: 'GNNBasedModel', optimizer, train_iterator, args, model2=None):
        """

        :param model:
        :param optimizer:
        :param train_iterator:
        :param args:
        :param model2: used for pseudo label predicting
        :return:
        """
        model.train()
        optimizer.zero_grad()

        t1 = time.time()

        positive_samples, negative_sample, subsampling_weight, query_graph, query_structures = next(
            train_iterator)

        # t2 = time.time()
        # print('loading ', t2 - t1)
        # t1 = t2

        if args.cuda:
            positive_samples = positive_samples.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            query_graph.to(model.device)

        with autocast(enabled=args.fp16):
            all_layer_pres = model(query_graph)

            last_layer_pre = all_layer_pres[-1]

            positive_logit, negative_logit = model.calc_logit(last_layer_pre, query_graph.target_node_idxes,
                                                              positive_samples=positive_samples,
                                                              negative_samples=negative_sample)

            log, loss = calc_loss(
                negative_logit, positive_logit, subsampling_weight)

        if model.fp16:
            model.scaler.scale(loss).backward()
            model.scaler.step(optimizer)
            model.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, mode='test'):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)
        layer_to_mrrs = [[] for _ in range(model.num_bi_dir_calibrates+1)]

        all_structure_to_weights = []

        with torch.no_grad():
            for negative_sample, query_graph, queries_unflatten, query_structures in tqdm(test_dataloader):
                if args.cuda:
                    negative_sample = negative_sample.cuda()
                    query_graph.to(model.device)

                all_layer_pres = model(query_graph)
                last_layer_pre = all_layer_pres[-1]

                min_query_idx = min(query_graph.query_idxes)
                # get the idx in current batch
                query_graph.query_idx_to_union_nodes = {k - min_query_idx: v for k, v in
                                                        query_graph.query_idx_to_union_nodes.items()}

                positive_logit, negative_logit = model.calc_logit(last_layer_pre, query_graph.target_node_idxes,
                                                                  query_idx_to_union_nodes=query_graph.query_idx_to_union_nodes,
                                                                  positive_samples=None,
                                                                  negative_samples=negative_sample)

                batch_logs, batch_mrrs = calc_metric(args, easy_answers, hard_answers, negative_logit,
                                                     queries_unflatten,
                                                     query_structures, last_layer_pre)

                layer_to_mrrs[-1].extend(batch_mrrs)

                for query_structure, res in batch_logs.items():
                    logs[query_structure].extend(res)

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' %
                                 (step, total_steps))

                step += 1

                if args.calc_all_layers_mrr and mode == 'test':
                    # record weights
                    for i, pres in enumerate(all_layer_pres[:-1]):
                        positive_logit, negative_logit = model.calc_logit(pres, query_graph.target_node_idxes,
                                                                          query_idx_to_union_nodes=query_graph.query_idx_to_union_nodes,
                                                                          positive_samples=None,
                                                                          negative_samples=negative_sample)

                        _, batch_mrrs = calc_metric(args, easy_answers, hard_answers, negative_logit,
                                                    queries_unflatten,
                                                    query_structures, pres)

                        layer_to_mrrs[i].extend(batch_mrrs)

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(
                logs[query_structure])

        if args.calc_all_layers_mrr and mode == 'test':
            # record weights
            layer_to_mrr = []
            for layer in range(len(layer_to_mrrs)):
                mrrs = layer_to_mrrs[layer]
                layer_to_mrr.append(sum(mrrs) / len(mrrs))
            metrics['layer_to_mrr'] = layer_to_mrr

        return metrics

    def sample_from_edge_type(self, edges, idxes):
        candidate_entities = []
        for edge in edges:
            candidate_entities.extend(self.edge_to_entities[edge])

        entities = [candidate_entities[idx %
                                       len(candidate_entities)] for idx in idxes]

        return entities


class RGAT(GNNBasedModel):
    def __init__(self, num_entities, num_relations, hidden_dim, edge_to_entities,
                 dropout=0.1, cuda=True, model_name=None, **kwargs):
        super(RGAT, self).__init__(num_entities, num_relations, hidden_dim, edge_to_entities, dropout, cuda, model_name,
                                   **kwargs)

        self.conv1 = RGATConv(
            hidden_dim, hidden_dim, num_relations=num_relations, edge_dim=hidden_dim, heads=1)
        self.conv2 = RGATConv(
            hidden_dim, hidden_dim, num_relations=num_relations, edge_dim=hidden_dim, heads=1)
        self.conv3 = RGATConv(
            hidden_dim, hidden_dim, num_relations=num_relations, edge_dim=hidden_dim, heads=1)

    def forward(self, graph):
        """
        node_to_entity: {node_idx: entity_idx}
        """

        # in P_net, node_to_entity will cover all nodes (if we use pseudo labels) except the masked (target) node
        node_to_entity = graph.node_to_entity
        operation_to_node = graph.operation_to_node
        num_nodes = graph.num_nodes
        edge_index = graph.edge_index
        edge_type = graph.edge_type

        # convert to bi-dir
        src, dst = edge_index
        src, dst = torch.cat((src, dst)), torch.cat((dst, src))
        edge_index = torch.stack((src, dst))
        edge_type = torch.cat((edge_type, get_inv_relation(edge_type)))

        t1 = time.time()

        x = torch.zeros((num_nodes, self.hidden_dim), device=self.device)
        # masked node and intermediate node (if pseudo labels are not used) are random initialized
        x = x.uniform_(-self.embedding_range.item(),
                       self.embedding_range.item())

        init_node_embeddings(self.entity_embedding, node_to_entity, x)

        # # using [MASK] token
        # x[other_node_idx] = self.entity_embedding.weight[-1]

        # t2 = time.time()
        # print('embedding ', t2 - t1)
        # t1 = t2

        # # use mean embedding of edge's targets to present variable node
        # # use the same idx to accelerate
        # batch_sample_nodes = []
        # batch_idxes = np.random.randint(low=0, high=self.num_entity, size=50 * len(other_node_idx)).reshape(-1, 50)
        #
        # for node_idx, idxes in zip(other_node_idx.tolist(), batch_idxes):
        #     in_edges = target_to_in_edges[node_idx]
        #     sample_nodes = self.sample_from_edge_type(in_edges, idxes)
        #     batch_sample_nodes.append(sample_nodes)

        # # (num_nodes, num_sample, hidden_dim) -> (num_nodes, hidden_dim)
        # mean_embedding = self.entity_embedding(
        #     torch.tensor(batch_sample_nodes, dtype=torch.long, device=self.device)).mean(axis=1)
        # x[other_node_idx] = mean_embedding

        edge_embedding = self.relation_embedding(edge_type)

        # t2 = time.time()
        # print('sampling ', t2 - t1)
        # t1 = t2

        x = F.dropout(
            self.conv1(
                x, edge_index, edge_type, edge_attr=edge_embedding).relu(),
            p=self.dropout_ratio, training=self.training)

        x = F.dropout(
            self.conv2(
                x, edge_index, edge_type, edge_attr=edge_embedding).relu(),
            p=self.dropout_ratio, training=self.training)

        x = self.conv3(x, edge_index, edge_type, edge_attr=edge_embedding)

        return x, None


def select_edges_by_nodes(edge_index, nodes, node_pos=0):
    # node_pos = 0: get edges starting from current node
    # node_pos = 1: get edges pointing to current node

    batch_edge_idx = []
    for node in nodes:
        edge_idxes = edge_index[node_pos] == node
        batch_edge_idx += [nonzero(edge_idxes).squeeze(-1)]
    batch_edge_idx = torch.cat(batch_edge_idx, dim=-1)

    return batch_edge_idx


class VecPredict(Predict):
    def __init__(self, num_entities, num_relations, hidden_dim, **kwargs):
        super().__init__(num_entities, num_relations, hidden_dim, **kwargs)
        # same as GQE
        # self.attn_lin = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

    def forward(self, x: Tensor, edge_index: Adj, edge_type: OptTensor = None,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x,
                             size=size, edge_attr=edge_attr)

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor, index) -> Tensor:
        # message form node j to node i (i.e. edge ji)
        message = x_j + edge_attr

        a = softmax(self.attn_lin(message), index)
        message = message * a

        return message


class VecGNN(GNNBasedModel):
    def __init__(self, num_entities, num_relations, hidden_dim, edge_to_entities,
                 dropout=0.1, cuda=True, model_name=None, **kwargs):
        super(VecGNN, self).__init__(num_entities, num_relations, hidden_dim, edge_to_entities, dropout, cuda,
                                     model_name, **kwargs)

        self.predict = VecPredict(
            num_entities, num_relations, hidden_dim, **kwargs)

    def forward(self, graph):
        """
        node_to_entity: {node_idx: entity_idx}
        """
        t1 = time.time()

        node_to_entity = graph.node_to_entity
        num_nodes = graph.num_nodes
        all_edge_types = graph.edge_type
        all_edges = graph.edge_index
        layer_to_edge_idxes: dict = graph.layer_to_edge_idxes

        anchor_node_idxes, anchor_node_entities = list(
            node_to_entity.keys()), list(node_to_entity.values())

        x = torch.zeros((num_nodes, self.hidden_dim), device=self.device)

        # init anchor node embedding
        anchor_node_entities = torch.tensor(
            anchor_node_entities, dtype=torch.long, device=self.device)
        anchor_node_idxes = torch.tensor(
            anchor_node_idxes, dtype=torch.long, device=self.device)

        x[anchor_node_idxes] = self.entity_embedding(anchor_node_entities)

        # t2 = time.time()
        # print('init ', t2 - t1)
        # t1 = t2

        # forward for each DAG layer
        for layer_idx in range(max(layer_to_edge_idxes.keys()) + 1):
            edge_idxes = layer_to_edge_idxes[layer_idx]
            edge_types = all_edge_types[edge_idxes]
            edges = all_edges[:, edge_idxes]
            target_idxes = torch.unique(all_edges[:, edge_idxes][1])
            relation_embedding = self.relation_embedding(edge_types)

            x[target_idxes] += self.predict(
                x, edges, edge_types, relation_embedding
            )[target_idxes]

        return x, None


class BiDAG(GNNBasedModel):
    def __init__(self, num_entities, num_relations, hidden_dim, edge_to_entities,
                 dropout=0.1, cuda=True, model_name=None, **kwargs):
        super(BiDAG, self).__init__(num_entities, num_relations, hidden_dim, edge_to_entities, dropout, cuda,
                                    model_name, **kwargs)
        self.num_bi_dir_calibrates = kwargs['num_bi_dir_calibrates']
        self.agg_method = kwargs['agg_method']
        self.num_heads = int(kwargs['num_heads'])

        Predicter, Calibrater = (GRUPredict, GRUCalibrate)

        self.predicter = Predicter(
            self.num_entity, self.num_relation, hidden_dim)

        # using predecessors and successors to calibrate
        self.bi_dir_cab_layers: ModuleList[Calibrater] = ModuleList([Calibrater(self.hidden_dim, num_heads=self.num_heads)
                                                                     for _ in range(self.num_bi_dir_calibrates)])

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        """
        node_to_entity: {node_idx: entity_idx}
        """
        t1 = time.time()

        node_to_entity = graph.node_to_entity
        num_nodes = graph.num_nodes
        all_edge_types = graph.edge_type
        all_edges = graph.edge_index
        layer_to_edge_idxes: dict = graph.layer_to_edge_idxes

        anchor_node_idxes, anchor_node_entities = list(
            node_to_entity.keys()), list(node_to_entity.values())

        x = [torch.zeros((num_nodes, self.hidden_dim), device=self.device)
             for _ in range(1 + self.num_bi_dir_calibrates)]

        # init anchor node embedding for each GNN layer
        anchor_node_entities = torch.tensor(
            anchor_node_entities, dtype=torch.long, device=self.device)
        anchor_node_idxes = torch.tensor(
            anchor_node_idxes, dtype=torch.long, device=self.device)

        x[0][anchor_node_idxes] = self.entity_embedding(anchor_node_entities)

        # t2 = time.time()
        # print('init ', t2 - t1)
        # t1 = t2

        # forward for each DAG layer
        for layer_idx in range(max(layer_to_edge_idxes.keys()) + 1):
            edge_idxes = layer_to_edge_idxes[layer_idx]
            edge_types = all_edge_types[edge_idxes]
            edges = all_edges[:, edge_idxes]
            target_idxes = torch.unique(all_edges[:, edge_idxes][1])
            relation_embedding = self.relation_embedding(edge_types)

            x[0][target_idxes] += self.predicter(
                x[0], edges, edge_types, relation_embedding
            )[target_idxes]

        current_layer_idx = 0
        if self.num_bi_dir_calibrates:
            # bi-dir correcting
            # create bi-directional graph first
            src, dst = all_edges
            src, dst = torch.cat((src, dst)), torch.cat((dst, src))
            all_bi_edges = torch.stack((src, dst))
            all_bi_edge_types = torch.cat(
                (all_edge_types, get_inv_relation(all_edge_types)))

            # step_norms = []
            for i, bi_dir_cab_layer in enumerate(self.bi_dir_cab_layers):
                step = bi_dir_cab_layer(
                    x[current_layer_idx], x[current_layer_idx],
                    all_bi_edges, all_bi_edge_types,
                    self.relation_embedding(all_bi_edge_types)
                )
                x[1 + current_layer_idx] = step + x[current_layer_idx]
                current_layer_idx += 1
                # step_norms.append(step.norm(dim=-1, p=2).mean())
            # print(step_norms)

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        glorot(m.weight)
        zeros(m.bias)


class GRUPredict(Predict):
    def __init__(self, num_entities, num_relations, hidden_dim, **kwargs):
        super().__init__(num_entities, num_relations, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: OptTensor = None,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x,
                             size=size, edge_attr=edge_attr)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        message = self.cell(x_j, edge_attr)

        return message

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        a = softmax(self.attn_lin(inputs), index)

        message = inputs * a

        return scatter(message, index, dim=0)


class GRUCalibrate(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_conv = AttnConv(hidden_dim, num_heads)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)

    def init_anchor_node(self, x):
        return self.cell(x)

    def forward(self, x_0, x_1, edges, edge_types, edge_attr, target_idxes=None):
        if target_idxes is None:
            target_idxes = list(range(x_0.shape[0]))

        attn_message = self.attn_conv(
            x_0, x_1,
            edges, edge_types,
            edge_attr
        )

        return self.cell(x_0[target_idxes], attn_message[target_idxes])


class AttnConv(MessagePassing):
    def __init__(self, hidden_dim, num_heads=1, **kwargs):
        super().__init__(node_dim=0, **kwargs)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.all_head_size = self.hidden_dim * num_heads
        self.negative_slope = 0.2

        self.Q = nn.Linear(hidden_dim, self.all_head_size)
        self.K = nn.Linear(hidden_dim * 2, self.all_head_size)
        self.V = nn.Linear(hidden_dim * 2, self.all_head_size)

        self.W = nn.Linear(self.all_head_size, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.Q, self.K, self.V, self.W]:
            glorot(item.weight)
            zeros(item.bias)

    def forward(self, x_0: Tensor, x_1: Tensor, edge_index: Adj, edge_type: OptTensor = None,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        """
        x_0: node embedding of last layer
        x_1: node embedding of current layer

        return
        - out : [num_edges, hidden_dim], attention message for target node
        """
        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x_0=x_0, x_1=x_1,
                             size=size, edge_attr=edge_attr)

        return out

    def message(self, x_0_i: Tensor, x_1_j: Tensor, edge_type: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        H, C = self.num_heads, self.hidden_dim

        q_i = self.Q(x_0_i).view(-1, H, C)
        k_j = self.K(torch.cat((edge_attr, x_1_j), dim=-1)).view(-1, H, C)
        v_j = self.V(torch.cat((edge_attr, x_1_j), dim=-1)).view(-1, H, C)

        # (num_edges, num_heads)
        alpha = torch.sum((q_i * k_j), dim=-1) / math.sqrt(self.hidden_dim)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = softmax(alpha, index, ptr, size_i)

        return alpha.view(-1, self.num_heads, 1) * v_j

    def update(self, aggr_out: Tensor) -> Tensor:
        aggr_out = aggr_out.view(-1, self.num_heads * self.hidden_dim)
        aggr_out = self.W(aggr_out)

        return aggr_out

    def __repr__(self) -> str:
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.hidden_dim, self.heads)
