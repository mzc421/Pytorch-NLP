# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
from config import parsers
# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。
# 所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码。
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch


def read_data(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签、句子的最大长度
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)
    # 根据不同的数据集返回不同的内容
    if os.path.split(file)[1] == "train.txt":
        max_len = max(max_length)
        return texts, labels, max_len
    return texts, labels,


class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.all_text = texts
        self.all_label = labels
        self.max_len = max_length
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)

    def __getitem__(self, index):
        # 取出一条数据并截断长度
        text = self.all_text[index][:self.max_len]
        label = self.all_label[index]

        # 分词
        text_id = self.tokenizer.tokenize(text)
        # 加上起始标志
        text_id = ["[CLS]"] + text_id

        # 编码
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        # 掩码  -》
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # 编码后  -》长度一致
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))
        # str -》 int
        label = int(label)

        # 转化成tensor
        token_ids = torch.tensor(token_ids)
        mask = torch.tensor(mask)
        label = torch.tensor(label)

        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label, max_len = read_data("./data/train.txt")
    print(train_text[0], train_label[0])
    trainDataset = MyDataset(train_text, train_label, max_len)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    for batch_text, batch_label in trainDataloader:
        print(batch_text, batch_label)
