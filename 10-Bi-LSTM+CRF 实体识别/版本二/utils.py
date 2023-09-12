# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
from torch.utils.data import Dataset, DataLoader
from config import parsers


def build_corpus(split, data_dir):
    """
    构建数据集
    :param split: 分割类型，是训练集，验证集 or 测试集
    :param data_dir: 数据路径
    :return:  word_lists(原始数据), tag_lists(标签数据), word2index, tag2index
    """
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
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 进行排序
    sorted_word_lists = sorted(word_lists, key=lambda x: len(x), reverse=False)
    sorted_tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=False)

    if split == "train":
        # 返回 word2index tag2index
        word_to_index = build_map(sorted_word_lists)
        tag_to_index = build_map(sorted_tag_lists)

        word_to_index['<PAD>'] = len(word_to_index)
        word_to_index['<UNK>'] = len(word_to_index)
        tag_to_index['<PAD>'] = len(tag_to_index)
        return word_lists, tag_lists, word_to_index, tag_to_index
    return word_lists, tag_lists


def build_map(lists):
    """
    构建 word2index和 tag2index 字典
    :param lists:
    :return: 返回一个字典
    """
    maps = {}
    for list_ in lists:
        for element in list_:
            if element not in maps:
                maps[element] = len(maps)
    return maps


class MyDataset(Dataset):
    """
    定义数据集对象
    """
    def __init__(self, datas, tags, word_2_index, tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word_2_index.get(i, self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]

        # tag_index = [self.tag_2_index[START_TAG]] + [self.tag_2_index[i] for i in tag] + [self.tag_2_index[STOP_TAG]]
        return data_index, tag_index

    def __len__(self):
        # 每句话的长度肯定和标签的长度一样
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):
        """
        这里是对每个 batch 做数据处理,进行数据的拼接
        这 batch_datas 长度是 batch_size 中规定的那个维度
        里面包含了 data 和 tag
        """

        datas, tags, batch_lens = [], [], []
        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        # 找出该批次的最大长度
        batch_max_len = max(batch_lens)

        datas = [i + [self.word_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag_2_index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas, dtype=torch.int64, device=self.device), \
               torch.tensor(tags, dtype=torch.long, device=self.device)


if __name__ == "__main__":
    args = parsers()
    train_data, train_tag, train_word2index, train_tag2index = build_corpus('train', args.train_file)

    tag_to_index = train_tag2index
    START_TAG, STOP_TAG = "<START>", "<STOP>"
    tag_to_index.update({START_TAG: len(tag_to_index), STOP_TAG: len(tag_to_index) + 1})

    train_dataset = MyDataset(train_data, train_tag, train_word2index, tag_to_index)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=train_dataset.pro_batch_data)

    for sentences, tags in train_loader:
        print("sentences.shape:", sentences.shape)
        print("tags.shape:", tags.shape)
        print("sentences[0].shape:", sentences[0].shape)
        print("tags[0].shape:", tags[0].shape)
        print("sentences[0]:", sentences[0])
        print("tags[0]:", tags[0])
        break
