# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import pickle as pkl
import numpy as np
import torch
from config import parsers
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


def read_data(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签
    texts, labels = [], []
    text_one, label_one = [], []
    for data in all_data:
        if data != "":
            text, label = data.split()
            text_one.append(text)
            label_one.append(label)
        else:
            texts.append(text_one)
            labels.append(label_one)
            text_one, label_one = [], []

    return texts, labels


def build_label_index(labels):
    label_to_index = {"PAD": 0, "UNK": 1}
    for label in labels:
        for i in label:
            if i not in label_to_index:
                label_to_index[i] = len(label_to_index)
    index_to_label = list(label_to_index)
    pkl.dump([label_to_index, index_to_label], open(parsers().data_pkl, "wb"))
    return label_to_index, index_to_label


class MyDataset(Dataset):
    def __init__(self, texts, label_to_index, with_labels=True, labels=None):
        self.all_text = texts
        self.all_label = labels
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
        self.label_index = label_to_index
        self.max_len = parsers().max_len

    def __getitem__(self, index):
        text = self.all_text[index]

        text_id = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_len + 2,
                                        padding="max_length", truncation=True, return_tensors="pt")
        text_id = text_id.squeeze(0)

        if self.with_labels:  # True if the dataset has labels
            label = self.all_label[index][:self.max_len]
            label_id = np.array([0] + [self.label_index.get(i, 1) for i in label] + [0] +
                                [0] * (self.max_len - len(text)))
            label_id = torch.tensor(label_id, dtype=torch.int64)
            return text_id, label_id
        else:
            return text_id

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


if __name__ == "__main__":
    args = parsers()
    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    test_text, test_label = read_data(args.test_file)
    label_index, index_label = build_label_index(train_label)

    trainDataset = MyDataset(train_text, label_index, labels=train_label, with_labels=True)
    trainLoader = DataLoader(trainDataset, batch_size=4, shuffle=False)
    for batch_text, batch_label in trainLoader:
        print(batch_text, batch_label)
        break
