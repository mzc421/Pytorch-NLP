# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from transformers import BertModel
import torch.nn as nn
from config import parsers
import torch
import torch.nn.functional as F


class TextCnnModel(nn.Module):
    def __init__(self):
        super(TextCnnModel, self).__init__()
        self.num_filter_total = parsers().num_filters * len(parsers().filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, parsers().class_num, bias=False)
        self.bias = nn.Parameter(torch.ones([parsers().class_num]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, parsers().num_filters, kernel_size=(size, parsers().hidden_size)) for size in parsers().filter_sizes
        ])

    def forward(self, x):
        # x: [batch_size, 12, hidden]
        x = x.unsqueeze(1)  # [batch_size, channel=1, 12, hidden]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            out = F.relu(conv(x))  # [batch_size, channel=2, 12-kernel_size[0]+1, 1]
            maxPool = nn.MaxPool2d(
                kernel_size=(parsers().encode_layer - parsers().filter_sizes[i] + 1, 1)
            )
            # maxPool: [batch_size, channel=2, 1, 1]
            pooled = maxPool(out).permute(0, 3, 2, 1)  # [batch_size, h=1, w=1, channel=2]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(parsers().filter_sizes))  # [batch_size, h=1, w=1, channel=2 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])  # [batch_size, 6]

        output = self.Weight(h_pool_flat) + self.bias  # [batch_size, class_num]

        return output


class BertTextModel_encode_layer(nn.Module):
    def __init__(self):
        super(BertTextModel_encode_layer, self).__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)

        for param in self.bert.parameters():
            param.requires_grad = True

        self.linear = nn.Linear(parsers().hidden_size, parsers().class_num)
        self.textCnn = TextCnnModel()

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True  # 确保 hidden_states 的输出有值
                            )
        # 取每一层encode出来的向量
        hidden_states = outputs.hidden_states  # 13 * [batch_size, seq_len, hidden] 第一层是 embedding 层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [batch_size, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textCnn的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [batch_size, 12, hidden]
        pred = self.textCnn(cls_embeddings)
        return pred


class BertTextModel_last_layer(nn.Module):
    def __init__(self):
        super(BertTextModel_last_layer, self).__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)
        for param in self.bert.parameters():
            param.requires_grad = True

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=parsers().num_filters, kernel_size=(k, parsers().hidden_size),) for k in parsers().filter_sizes]
        )
        self.dropout = nn.Dropout(parsers().dropout)
        self.fc = nn.Linear(parsers().num_filters * len(parsers().filter_sizes), parsers().class_num)

    def conv_pool(self, x, conv):
        x = conv(x)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1, 1]
        x = F.relu(x)
        x = x.squeeze(3)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1]
        size = x.size(2)
        x = F.max_pool1d(x, size)   # shape[batch+size, out_channels, 1]
        x = x.squeeze(2)  # shape[batch+size, out_channels]
        return x

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]  # shape [batch_size, max_len]
        hidden_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               output_hidden_states=False)
        out = hidden_out.last_hidden_state.unsqueeze(1)   # shape [batch_size, 1, max_len, hidden_size]
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs], 1)  # shape  [batch_size, parsers().num_filters * len(parsers().filter_sizes]
        out = self.dropout(out)
        out = self.fc(out)
        return out


