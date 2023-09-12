# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_ATT(nn.Module):
    def __init__(self, config):
        super(BiLSTM_ATT, self).__init__()
        self.batch = config.batch_size
        self.embedding_size = config.embedding_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.tag_size = config.tag_size
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim
        self.device = config.device

        # 创建一个词嵌入层
        self.word_embeds = nn.Embedding(self.embedding_size, self.embedding_dim).to(self.device)
        # 创建位置嵌入层
        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim).to(self.device)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim).to(self.device)
        # 创建关系嵌入层
        self.relation_embeds = nn.Embedding(self.tag_size, self.hidden_dim).to(self.device)

        # 创建双向LSTM层
        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim * 2, hidden_size=self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, device=self.device)

        # 创建线性层，用于将LSTM输出映射到标签空间
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

        # 创建Dropout层，用于正则化
        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)

        # 初始化LSTM的隐藏状态
        self.hidden = self.init_hidden()

        # 创建注意力权重和关系偏置
        self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.hidden_dim)).to(self.device)
        self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1)).to(self.device)

    def init_hidden(self):
        """
        初始化LSTM的隐藏状态
        """
        return torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device)

    def init_hidden_lstm(self):
        """
        初始化LSTM层的隐藏状态和记忆单元
        """
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device))

    def attention(self, H):
        """
        定义注意力机制函数
        """
        # 进行非线性变换
        M = torch.tanh(H)
        # 进行批次的矩阵乘法，计算注意力权重。注意力权重的计算包含两个部分：self.att_weight是表示注意力权重的参数矩阵，M是经过非线性变换后的张量
        # 对于 torch.bmm()函数介绍可以看README.md文档
        a = F.softmax(torch.bmm(self.att_weight, M), 2)
        # 交换注意力权重张量a的维度，使得注意力权重与输入张量H的维度相匹配
        a = torch.transpose(a, 1, 2)
        # 再次进行批次的矩阵乘法，计算注意力加权后的结果
        attention_result = torch.bmm(H, a)
        # 返回注意力加权后的结果
        return attention_result

    def forward(self, sentence, pos1, pos2):
        """
        前向传播函数
        """
        # sentence/pos1/pos2 [batch_size, max_len]
        self.hidden = self.init_hidden_lstm()
        # 将词嵌入、位置嵌入拼接成一个输入张量
        # [batch_size, max_len, 100 + 25 + 25]
        embeds = torch.cat((self.word_embeds(sentence), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), 2)

        # 转置输入张量的维度
        # [max_len, batch_size, 100 + 25 + 25]
        embeds = torch.transpose(embeds, 0, 1)

        # 经过LSTM层
        # [max_len, batch_size, 100*2]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 维度变换
        lstm_out = torch.transpose(lstm_out, 0, 1)
        # [batch_size, 100*2, max_len]
        lstm_out = torch.transpose(lstm_out, 1, 2)

        # 应用Dropout
        # [batch_size, 100*2, max_len]
        lstm_out = self.dropout_lstm(lstm_out)

        # 应用注意力机制
        # [batch_size, 100*2, 1]
        att_out = torch.tanh(self.attention(lstm_out))
        # att_out = self.dropout_att(att_out)

        # 映射关系标签
        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1).to(self.device)
        # [batch_size, 12, 100*2]
        relation = self.relation_embeds(relation)

        # 计算最终的输出
        # [batch_size, 12, 1]
        res = torch.add(torch.bmm(relation, att_out.to(self.device)), self.relation_bias)
        # [batch_size, 12, 1]
        res = F.softmax(res, 1)
        return res.view(self.batch, -1)
