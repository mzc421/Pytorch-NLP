# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch.nn as nn
from transformers import BertModel
from config import parsers
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.args = parsers()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 加载 bert 中文预训练模型
        self.bert = BertModel.from_pretrained(self.args.bert_pred)
        # 让 bert 模型进行微调（参数在训练过程中变化）
        for param in self.bert.parameters():
            param.requires_grad = True
        # 全连接层
        self.linear = nn.Linear(self.args.num_filters, self.args.class_num)

    def forward(self, x):
        input_ids, attention_mask = x[0].to(self.device), x[1].to(self.device)
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)  # 控制是否输出所有encoder层的结果
        # shape (batch_size, hidden_size)  pooler_output -->  hidden_out[0]
        pred = self.linear(hidden_out.pooler_output)
        # 返回预测结果
        return pred

# bert的输出结果有四个维度： last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。
# pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。
# （通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）

# hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(
# batch_size, sequence_length, hidden_size)
# attentions：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True, 它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。

# cross_attentions：shape是(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)

# 我们是微调模式，需要获取bert最后一个隐藏层的输出输入到下一个全连接层，所以取第一个维度，也就是hiden_outputs[0]
