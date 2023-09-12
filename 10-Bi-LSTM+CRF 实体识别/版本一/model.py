# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, pad_index):
        """
        vocab_size: word_to_index 词典数量
        tag_to_ix：tag_to_index 词典内容
        embedding_dim：Embedding 层数（维数）
        hidden_dim：Bi-LSTM 隐藏层层数
        pad_index：<[PAD]> 在 word_to_index 词典中的位置
        batch_size：批次数
        """
        super(BiLSTM_CRF, self).__init__()
        self.hidden = None
        self.batch_size = None
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)
        self.pad_idx = pad_index
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 中间层设置
        # embedding 层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_idx)  # 转词向量
        # Bi-Lstm 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # Bi-LSTM 的输出对应 tag 空间（tag space）
        # 输入是 [batch_size, size] 中的 size，输出是 [batch_size，output_size] 的 output_size
        self.linear = nn.Linear(hidden_dim, self.target_size)
        # CRF层
        self.crf = CRF(self.target_size, batch_first=True)  # batch_size 默认为 False

    def forward(self, sentence, tags=None, mask=None):
        # sentence=(batch, seq_len)   tags=(batch, seq_len)  masks=(batch, seq_len)
        # 1. 从 sentence 到 Embedding 层
        embeds = self.word_embeds(sentence).permute(1, 0, 2)  # shape [seq_len, batch_size, embedding_size]

        # 2. 从 Embedding 层到 Bi-LSTM 层
        # Bi-lstm 层的隐藏节点设置
        # 隐藏层就是（h_0, c_0）    num_directions = 2 if self.bidirectional else 1
        # h_0 的结构：(num_layers*num_directions, batch_size, hidden_size)
        self.hidden = (torch.randn(2, sentence.shape[0], self.hidden_dim // 2, device=self.device),
                       torch.randn(2, sentence.shape[0], self.hidden_dim // 2, device=self.device))

        # input=(seq_length, batch_size, embedding_num)
        # output(lstm_out)=(seq_length, batch_size, num_directions * hidden_size)
        # h_0 = (num_layers*num_directions, batch_size, hidden_size)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # 3. 从 Bi-LSTM 层到全连接层
        # 从 Bi-lstm 的输出转为 target_size 长度的向量组（即输出了每个 tag 的可能性）
        # 输出 shape=(seq_length, batch_size, len(tag_to_ix))
        lstm_feats = self.linear(lstm_out)

        # 4. 全连接层到 CRF 层
        if tags is not None:
            # 训练用
            if mask is not None:
                loss = -1. * self.crf(emissions=lstm_feats.permute(1, 0, 2), tags=tags, mask=mask, reduction='mean')
                # outputs=(batch_size,)   输出 log 形式的 likelihood
            else:
                loss = -1. * self.crf(emissions=lstm_feats.permute(1, 0, 2), tags=tags, reduction='mean')
            return loss
        else:
            # 测试
            if mask is not None:
                prediction = self.crf.decode(emissions=lstm_feats.permute(1, 0, 2), mask=mask)
            else:
                prediction = self.crf.decode(emissions=lstm_feats.permute(1, 0, 2))
            return prediction
