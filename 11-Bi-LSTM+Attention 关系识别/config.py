# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--all_data", type=str, default='./data/all_data.txt')
parser.add_argument("--target_file", type=str, default='./data/relation2id.txt')
parser.add_argument("--param_dict", type=str, default='./data/param_dict.txt')
parser.add_argument("--model_best", type=str, default='./model/model_best.pt')
parser.add_argument("--model_last", type=str, default='./model/model_last.pt')

# model
# 关系标签的数量。这个超参数定义了模型要预测的关系类别的数量，通常对应于任务中的不同关系类型。
parser.add_argument('--tag_size', type=int, default=12, help='The number of relation labels. This hyperparameter defines the number of different relation categories that the model needs to predict, typically corresponding to the types of relationships in the task.')
# 位置嵌入的维度。这指定了用于表示实体位置信息的嵌入向量的维度。
parser.add_argument("--pos_size", type=int, default=82, help="The dimensionality of position embeddings. This specifies the dimension of the embedding vectors used to represent entity position information.")
# 位置嵌入的大小。这是位置嵌入向量的维度，决定了模型如何利用实体的位置信息。
parser.add_argument("--pos_dim", type=int, default=25, help="The size of position embeddings. This is the dimensionality of position embedding vectors, determining how the model utilizes entity position information.")
# 词嵌入矩阵的大小。它决定了词嵌入的维度，即每个词语将被表示为一个具有多少个特征的向量。
parser.add_argument('--embedding_size', type=int, default=768, help='The size of the embedding matrix, which defines the dimensionality of word embeddings. It specifies how many features are used to represent each word.')
# 词嵌入维度。它表示每个词嵌入向量的维度，影响了模型对词语语义信息的捕获能力。
parser.add_argument('--embedding_dim', type=int, default=100, help="The dimensionality of word embeddings, representing the number of features in each word embedding vector. It affects the model's ability to capture semantic information of words.")
# 隐藏层维度。这是LSTM层和其他隐藏层的维度大小，控制了模型的复杂度和学习能力。
parser.add_argument('--hidden_dim', type=int, default=200, help="The dimensionality of hidden layers, including LSTM layers and other hidden layers. It controls the model's complexity and learning capacity.")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.0001)
hparams = parser.parse_args()
