# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai


import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, corpus_num, class_num, embedding_num, hidden_num, bi=True):
        super().__init__()

        self.pred = None
        self.embedding = nn.Embedding(corpus_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num, batch_first=True, bidirectional=bi)

        if bi:
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, text, label=None):
        embedding = self.embedding(text)
        out, _ = self.lstm(embedding)
        pred = self.classifier(out)
        self.pred = torch.argmax(pred, dim=-1).reshape(-1)

        if label is not None:
            loss = self.loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            return loss
        return torch.argmax(pred, dim=-1).reshape(-1)

