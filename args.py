import argparse
import datetime
import json
import logging
import os

from utils.util import parse_time


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int,
                        help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=400, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=50, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int,
                        help="no need to set manually, will configure automatically")

    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--num_entities', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_relations', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--geo', default='vec', type=str,
                        help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=3407, type=int, help="random seed")
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument('--agg_method', type=str, default='gru', help='method in GNN message aggregating')
    parser.add_argument('--num_bi_dir_calibrates', type=int, default=0, help='number of bi-dir calibrating layers')
    parser.add_argument('--num_heads', default=6)
    parser.add_argument('--warm_up', type=float, default=0.2)

    parser.add_argument('--calc_all_layers_mrr', action='store_true')
    
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args(args)
    args = post_init_args(args)

    return args


def post_init_args(args):
    """
    post init args, set save_path and so on.
    """

    tasks = args.tasks.split('.')
    ori_task = args.tasks
    args.tasks = tasks
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    prefix = 'logs' if args.prefix is None else args.prefix
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], ori_task, args.geo)

    if args.checkpoint_path is not None:
        if not args.only_load_pretrained:
            # if only_load_pretrained, new log will be created
            args.save_path = args.checkpoint_path

        config_path = os.path.join(args.checkpoint_path, 'config.json')
        print(f'Loading config from {config_path}...')
        with open(config_path, 'r') as f:
            config = json.load(f)
        args.hidden_dim = config['hidden_dim']
        args.gamma = config['gamma']
    else:
        if args.geo in ['box']:
            tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
        elif args.geo in ['vec']:
            tmp_str = "g-{}".format(args.gamma)
        elif args.geo == 'beta':
            tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
        else:
            tmp_str = args.geo

        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # set logger after setting save_path
    set_logger(args)

    logging.info(f'logging to {args.save_path}')

    # set num_entities and num_relations
    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        num_entities = int(entrel[0].split(' ')[-1])
        num_relations = int(entrel[1].split(' ')[-1])

    # contain inverse relation already
    args.num_entities = num_entities
    args.num_relations = num_relations

    with open(args.save_path + '/config.json', 'w') as f:
        json.dump(vars(args), f)

    return args


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    # def beijing(sec, what):
    #     beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    #     return beijing_time.timetuple()
    #
    # logging.Formatter.converter = beijing

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
