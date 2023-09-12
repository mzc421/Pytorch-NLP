# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai


import torch
from torch.utils.data import Dataset, DataLoader
from config import parsers


def build_corpus(split, data_dir, make_vocab=True):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list + ["<END>"])
                tag_lists.append(tag_list + ["<END>"])

                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    # 如果make_vocab为True，还需要返回word2id和tag_2_index
    if make_vocab:
        word2id = bulid_word_tag_index(word_lists)
        tag_2_index = bulid_word_tag_index(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)
        word2id["<START>"] = len(word2id)
        # word2id["<END>"]   = len(word2id)

        tag_2_index['<PAD>'] = len(tag_2_index)
        tag_2_index["<START>"] = len(tag_2_index)
        # tag_2_index["<END>"] = len(tag_2_index)
        return word_lists, tag_lists, word2id, tag_2_index
    else:
        return word_lists, tag_lists


def bulid_word_tag_index(datas):
    maps_index = {}
    for data in datas:
        for word in data:
            if word not in maps_index:
                maps_index[word] = len(maps_index)
    return maps_index


class MyDataset(Dataset):
    def __init__(self, datas, tags, word_index, tag_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_index
        self.tag_2_index = tag_index
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word_2_index.get(i, self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]

        return data_index, tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):
        """自定义数据批次长度处理"""
        datas, tags, masks = [], [], []

        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            masks.append(len(data))
        batch_max_len = max(masks)

        datas_id = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        masks_id = [[1] * len(i) + [0] * (batch_max_len - len(i)) for i in datas]
        tags_id = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas_id, dtype=torch.int64, device=self.device), \
               torch.tensor(tags_id, dtype=torch.long, device=self.device),\
               torch.tensor(masks_id, dtype=torch.bool, device=self.device)


if __name__ == "__main__":
    args = parsers()

    train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", args.train_file, make_vocab=True)
    print('tag_2_index: ', tag_2_index)

    train_dataset = MyDataset(train_data, train_tag, word_2_index, tag_2_index)
    train_loader = DataLoader(train_dataset, 2, shuffle=False, collate_fn=train_dataset.pro_batch_data)
    for batch_data, batch_tag, batch_mask in train_loader:
        print("batch_data.shape:", batch_data.shape)
        print("batch_tag.shape:", batch_tag.shape)
        print("batch_mask.shape:", batch_mask.shape)
        print("*"*100)
        print("batch_data[2].shape:", batch_data[1].shape)
        print("batch_tag[2].shape:", batch_tag[1].shape)
        print("batch_mask[2].shape:", batch_mask[1].shape)
        print("*"*100)
        print("batch_data[2]:", batch_data[0])
        print("batch_tag[2]:", batch_tag[0])
        print("batch_mask[2]:", batch_mask[0])
        break
