import itertools
import json
import logging

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from args import parse_args
from models import KGReasoning
from gnn_models import RGAT, BiDAG, VecGNN
from data import TestDataset, TrainDataset, SingledirectionalOneShotIterator, DagTrainDataset, DagTestDataset, \
    query_name_dict, load_data
from tensorboardX import SummaryWriter

from collections import defaultdict

from utils import flatten_query, set_global_seed, eval_tuple, save_model, log_metrics, get_mean_val_of_dicts

import csv


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(
        model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode)
    num_query_structures = 0
    num_queries = 0

    if args.calc_all_layers_mrr and mode == 'test':
        layer_to_mrr = metrics.pop('layer_to_mrr')
        file_path = os.path.join(args.save_path, f'{mode}_all_layer_mrrs')
        with open(file_path, 'a+') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerow(layer_to_mrr)

    for query_structure in metrics:
        log_metrics(
            mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]),
                              metrics[query_structure][metric], step)
            all_metrics["_".join(
                [query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar(
            "_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)

    return all_metrics


def main(args):
    set_global_seed(args.seed)

    tasks, num_entities, num_relations = args.tasks, args.num_entities, args.num_relations

    TrainDataset_, TestDataset_ = DagTrainDataset, DagTestDataset

    if not args.do_train:  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)

    # construct edge to tail entities
    logging.info(f'constructing s to po')
    df = pd.read_csv('%s/train.txt' % args.data_path,
                     sep='\t', names=['h', 'r', 't'])
    edge_to_entities = defaultdict(list)
    for r, t in zip(df.r.values, df.t.values):
        edge_to_entities[r].append(t)

    logging.info('-------------------------------' * 3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % num_entities)
    logging.info('#relation: %d' % num_relations)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data(
        args, tasks)

    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure] +
                         ": " + str(len(train_queries[query_structure])))

        train_queries = flatten_query(train_queries)

        train_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset_(train_queries, num_entities, num_relations, args.negative_sample_size, train_answers,
                          args.data_path),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset_.collate_fn,
        ))

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure] +
                         ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset_(
                valid_queries,
                args.num_entities,
                args.num_relations,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset_.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure] +
                         ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset_(
                test_queries,
                args.num_entities,
                args.num_relations,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset_.collate_fn
        )

    name_to_model = {
        'gnn': RGAT,
        'dagnn': BiDAG,
    }

    if args.checkpoint_path is not None:
        with open(os.path.join(args.checkpoint_path, 'config.json')) as f:
            kwargs = json.load(f)
    else:
        kwargs = vars(args)

    model = name_to_model[args.geo](
        edge_to_entities=edge_to_entities,
        **kwargs
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        logging.info('moving model to cuda...')
        model.cuda()

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        warm_up_steps = int(args.max_steps * args.warm_up)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=args.max_steps,
        )

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(
            args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []

        for step in range(init_step, args.max_steps):
            log = model.train_step(
                model, optimizer, train_iterator, args)

            for metric in log:
                writer.add_scalar('path_' + metric, log[metric], step)

            training_logs.append(log)
            lr_scheduler.step()

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args,
                                                 valid_dataloader,
                                                 query_name_dict, 'valid', step, writer)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                                query_name_dict, 'test', step, writer)

                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list,
                           args, lr_scheduler)

            if step % args.log_steps == 0 and step > 0:
                metrics = {}
                
                for metric in training_logs[0].keys():
                    metrics[metric] = sum(
                        [log[metric] for log in training_logs]) / len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args, lr_scheduler)

    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                    query_name_dict,
                                    'test', step, writer)

    logging.info("Training finished!!")


if __name__ == '__main__':
    main(parse_args())
