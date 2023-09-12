# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import argparse


def parsers():
    parser = argparse.ArgumentParser(description="BiLSTM entity recognition of argparse")
    parser.add_argument('--train_file', type=str, default=os.path.join("data", "train.char.bmes"))
    parser.add_argument('--test_file', type=str, default=os.path.join("data", "test.char.bmes"))
    parser.add_argument('--dev_file', type=str, default=os.path.join("data", "dev.char.bmes"))
    parser.add_argument('--data_pkl', type=str, default=os.path.join("data", "data_parameter.pkl"))
    parser.add_argument('--epochs', type=int, default=230)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_num', type=int, default=101)
    parser.add_argument('--hidden_num', type=int, default=107)
    parser.add_argument('--learn_rate', type=float, default=1e-4)
    parser.add_argument('--bi', type=bool, default=False)
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args
