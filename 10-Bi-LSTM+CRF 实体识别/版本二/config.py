# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import argparse


def parsers():
    parser = argparse.ArgumentParser(description="BiLSTM + CRF entity recognition of argparse")
    parser.add_argument('--train_file', type=str, default=os.path.join("./data", "train.txt"))
    parser.add_argument('--test_file', type=str, default=os.path.join("./data", "test.txt"))
    parser.add_argument('--dev_file', type=str, default=os.path.join("./data", "dev.txt"))
    parser.add_argument('--data_pkl', type=str, default=os.path.join("./data", "data_pkl.pkl"))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--dev_test_batch_size', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--learn_rate', type=float, default=1e-2)
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args
