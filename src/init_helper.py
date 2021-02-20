import argparse
import logging
import random

import numpy as np
import torch


def init_logger(log_path):
    logger = logging.getLogger()
    format_str = '[%(asctime)s %(filename)s#%(lineno)3d] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(format_str))
        logger.addHandler(fh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume-step', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--encoder', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--log-step', type=int, default=10)
    parser.add_argument('--save-step', type=int, default=4750)
    parser.add_argument('--matching', type=str, choices=['conv', 'cosine'], default='conv')
    parser.add_argument('--self-structure', type=str, choices=['cluster', 'basic'], default='cluster')
    parser.add_argument('--keep-topk', type=int, default=32)
    parser.add_argument('--input-height', type=int, default=512)
    parser.add_argument('--input-width', type=int, default=512)

    # optimizer
    parser.add_argument('--base-lr', type=float, default=1e-5)
    parser.add_argument('--encoder-lr-weight', type=float, default=1.0)
    parser.add_argument('--lr-decay-step', type=int, default=9501)
    parser.add_argument('--lr-decay-rate', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)

    # augmentation
    parser.add_argument('--flip-rate', type=float, default=0.5)
    parser.add_argument('--blur-rate', type=float, default=1.0)
    parser.add_argument('--rotation', type=int, default=45)
    parser.add_argument('--min-margin', type=float, default=0.0)
    parser.add_argument('--max-margin', type=float, default=1.0)
    parser.add_argument('--fix-margin', type=float, default=0.5)

    # output
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--save-dir', type=str, default=None)

    # dataset
    parser.add_argument('--split', type=str, required=True)

    args = parser.parse_args()
    return args
