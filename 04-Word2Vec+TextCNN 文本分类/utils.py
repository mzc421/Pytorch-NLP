# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import gensim
import numpy as np
from gensim.models import Word2Vec
from torch.utils.data import Dataset
import torch
from config import parsers
import jieba


np.random.seed(0)


def read_data(file):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts, labels = [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            text = [i for i in jieba.cut(text) if i != " "]
            texts.append(text)
            labels.append(label)
    return texts, labels


def built_curpus(train_texts, embedding_num):
    texts = [[i for i in text if i != " "] for text in train_texts]
    model = Word2Vec(texts, epochs=10, sg=0, vector_size=embedding_num, window=3, min_count=1)
    model.wv.save_word2vec_format(parsers().data_words_model, binary=True)


class TextDataset(Dataset):
    def __init__(self, all_text, max_len, with_labels=True, all_label=None):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(parsers().data_model, binary=True)

        self.all_text = all_text
        self.all_label = all_label
        self.max_len = max_len
        self.num_word = self.model.vectors.shape[0]
        self.embedding = self.model.vectors.shape[1]

        self.with_labels = with_labels

        # 添加 "<pad>" 和 "<UNK>"
        # {"<PAD>": np.zeros(self.embedding), "<UNK>": np.random.randn(self.embedding)}
        self.Embedding = self.model.vectors
        self.Embedding = np.insert(self.Embedding, self.num_word, [np.zeros(self.embedding), np.random.randn(self.embedding)], axis=0)

        self.word_2_index = self.model.key_to_index
        self.word_2_index.update({"<PAD>": self.num_word, "<UNK>": self.num_word + 1})

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]

        text_id = [self.word_2_index.get(i, self.word_2_index["<UNK>"]) for i in text]
        text_id = text_id + [self.word_2_index["<PAD>"]] * (self.max_len - len(text_id))

        wordEmbedding = np.array([self.Embedding[i] for i in text_id])

        text_id = torch.tensor(wordEmbedding).unsqueeze(dim=0)

        if self.with_labels:
            label = int(self.all_label[index])
            return text_id, label
        else:
            return text_id

    def __len__(self):
        return len(self.all_text)
