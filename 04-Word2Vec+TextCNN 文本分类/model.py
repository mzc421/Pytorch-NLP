# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, kernel_s, embeddin_num, max_len, hidden_num):
        super().__init__()
        # shape [batch *  in_channel * max_len * emb_num]
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embeddin_num))
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):  # shape [batch *  in_channel * max_len * emb_num]
        c = self.cnn(batch_emb)
        a = self.act(c)
        a = a.squeeze(dim=-1)
        m = self.mxp(a)
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self, embedding_num, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = embedding_num

        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)

        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 2 * 3
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx):   # shape torch.Size([batch_size, 1, max_len, embedding])
        b1_result = self.block1(batch_idx)  # shape torch.Size([batch_size, 2])
        b2_result = self.block2(batch_idx)  # shape torch.Size([batch_size, 2])
        b3_result = self.block3(batch_idx)  # shape torch.Size([batch_size, 2])

        # 拼接
        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)  # shape torch.Size([batch_size, 6])
        pre = self.classifier(feature)  # shape torch.Size([batch_size, class_num])

        return pre

