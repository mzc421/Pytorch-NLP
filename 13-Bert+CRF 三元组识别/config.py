# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import argparse


def parsers():
    parser = argparse.ArgumentParser(description="Identification of triples of argparse")
    parser.add_argument('--all_data', type=str, default=os.path.join("./data", "all_data.txt"))
    parser.add_argument('--train_file', type=str, default=os.path.join("./data", "train.json"))
    parser.add_argument('--dev_file', type=str, default=os.path.join("./data", "val.json"))
    parser.add_argument('--test_file', type=str, default=os.path.join("./data", "test.json"))
    parser.add_argument('--bert_file', type=str, default="./bert-base-chinese")
    parser.add_argument('--relation', type=str, default=os.path.join("./data", "relation.txt"))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learn_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--save_pretrained_best", type=str, default=os.path.join("model", 'saved_best_model'))
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_pretrained_last", type=str, default=os.path.join("model", 'saved_last_model'))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args
