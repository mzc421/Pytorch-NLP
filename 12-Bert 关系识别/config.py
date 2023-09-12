# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--bert_path", type=str, default="./bert-base-chinese")
parser.add_argument("--all_data", type=str, default='./data/all_data.txt')
parser.add_argument("--train_file", type=str, default='./data/train.json')
parser.add_argument("--dev_file", type=str, default='./data/val.json')
parser.add_argument("--test_file", type=str, default='./data/test.json')
parser.add_argument("--target_file", type=str, default='./data/relation.txt')
parser.add_argument("--log_dir", type=str, default='log')
parser.add_argument("--model_best_bin", type=str, default='./model/model_best.bin')
parser.add_argument("--model_best_checkpoint_bin", type=str, default='./model/checkpoint_best.bin')
parser.add_argument("--model_last_bin", type=str, default='./model/model_last.bin')
parser.add_argument("--model_last_checkpoint_bin", type=str, default='./model/checkpoint_last.bin')

# model
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument('--dropout', type=float, default=0.1, required=False, help='dropout')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=0)
hparams = parser.parse_args()
