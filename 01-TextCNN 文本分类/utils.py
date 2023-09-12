# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch.nn as nn
from torch.utils.data import Dataset
import torch
import pickle as pkl
from config import parsers


def read_data(file):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts, labels = [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            texts.append(text)
            labels.append(label)
    return texts, labels


def built_curpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    embedding = nn.Embedding(len(word_2_index), embedding_num)
    pkl.dump([word_2_index, embedding], open(parsers().data_pkl, "wb"))
    return word_2_index, embedding


class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(i, 1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return text_idx, label

    def __len__(self):
        return len(self.all_text)
