# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from config import parsers


def read_data(file, mode="train"):
    with open(file, "r", encoding="utf-8") as f:
        all_datas = f.read().split("\n")

    all_text, text, all_label, label = [], [], [], []
    for data in all_datas:
        if data == "":
            all_text.append(text)
            all_label.append(label)
            text, label = [], []
        else:
            text.append(data.split()[0])
            label.append(data.split()[1])

    all_text = sorted(all_text, key=lambda x: len(x), reverse=False)
    all_label = sorted(all_label, key=lambda x: len(x), reverse=False)

    if mode == "train":
        word_index, label_index, index_label = build_map(all_text, all_label)
        return all_text, all_label, word_index, label_index, index_label

    return all_text, all_label


def build_map(texts, labels):
    word_index, label_index = {}, {}
    for text, label in zip(texts, labels):
        for i, j in zip(text, label):
            if i not in word_index:
                word_index[i] = len(word_index)
            if j not in label_index:
                label_index[j] = len(label_index)

    word_index['<UNK>'] = len(word_index)
    word_index['<PAD>'] = len(word_index)
    label_index['<PAD>'] = len(label_index)
    return word_index, label_index, [i for i in label_index]


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, word_index, label_index):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.label_index = label_index

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        text_id = [self.word_index.get(i, self.word_index["<UNK>"]) for i in text]
        label_id = [self.label_index[i] for i in label]

        return text_id, label_id

    def __len__(self):
        return self.texts.__len__()

    def pro_batch_data(self, batch_data):
        texts, labels, batch_len = [], [], []
        for i in batch_data:
            texts.append(i[0])
            labels.append(i[1])
            batch_len.append(len(i[0]))

        max_batch_len = max(batch_len)

        texts = [i + [self.word_index["<PAD>"]] * (max_batch_len - len(i)) for i in texts]
        labels = [i + [self.label_index["<PAD>"]] * (max_batch_len - len(i)) for i in labels]

        texts = torch.tensor(texts, dtype=torch.int64, device="cuda:0" if torch.cuda.is_available() else "cpu")
        labels = torch.tensor(labels, dtype=torch.long, device="cuda:0" if torch.cuda.is_available() else "cpu")

        return texts, labels


def prepare_data():
    args = parsers()
    train_text, train_label, word_index, label_index, index_label = read_data(args.train_file, mode="train")
    dev_text, dev_label = read_data(args.dev_file, mode="dev")
    test_text, test_label = read_data(args.test_file, mode="test")

    # 所有不重复的汉字
    corpus_num = len(word_index)
    # 所有类别
    class_num = len(label_index)

    train_dataset = BiLSTMDataset(test_text, test_label, word_index, label_index)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)
    dev_dataset = BiLSTMDataset(dev_text, dev_label, word_index, label_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=train_dataset.pro_batch_data)
    test_dataset = BiLSTMDataset(test_text, test_label, word_index, label_index)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=train_dataset.pro_batch_data)

    pkl.dump([word_index, label_index, index_label, corpus_num, class_num], open(args.data_pkl, "wb"))

    return train_dataloader, dev_dataloader, test_dataloader, index_label, corpus_num, class_num


if __name__ == "__main__":
    args = parsers()
    train_loader, dev_loader, test_loader, index_label, corpus_num, class_num = prepare_data()

    for batch_data, batch_label in train_loader:
        print(batch_data, batch_label)
        break
