# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai


from transformers import BertModel
import torch.nn as nn
from config import parsers


class BertNerModel(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)

        for name, param in self.bert.named_parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(768, class_num)

    def forward(self, batch_index):
        # bert_out0:字符级别特征 [batch_size, max_len+2, 768]
        # bert_out1:篇章级别  [batch_size, 768]
        bert_out = self.bert(batch_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]
        pre = self.classifier(bert_out0)
        return pre


