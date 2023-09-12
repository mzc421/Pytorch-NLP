# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import re
import os
import json
from config import hparams
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence


class MyTokenizer(object):
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
        self.mask_entity = mask_entity

    def tokenize(self, item):
        sentence, pos_head, pos_tail = item['text'], item['h']['pos'], item['t']['pos']

        # 将文本中第一个出现的实体作为实体1，第二个出现的作为实体2
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        # 切词
        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])

        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])

        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])

        # 合并
        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]

        re_tokens = ['[CLS]']
        cur_pos = 0  # 索引
        # 两个实体的位置
        pos1, pos2 = [0, 0], [0, 0]
        for token in tokens:
            # 实体1与实体2 的首个位置
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')

            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)

            # 实体1与实体2 的末尾位置
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)

            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)

            cur_pos += 1

        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2


def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def read_data(input_file, target_path, tokenizer=None):
    data_set = []
    tag2idx = get_tag2idx(target_path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()
    for line in tqdm(data_lines, desc=f"处理数据 {os.path.split(input_file)[-1][:-5]}"):
        item = json.loads(line.strip())
        token, pos_e1, pos_e2 = tokenizer.tokenize(item)

        max_len = max(len(token), pos_e1[1], pos_e2[1])
        e1_mask = convert_pos_to_mask(pos_e1, max_len)
        e2_mask = convert_pos_to_mask(pos_e2, max_len)

        encoded = tokenizer.bert_tokenizer.encode_plus(token, max_length=max_len, truncation=True)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']
        attention_mask = encoded['attention_mask']

        data_set.append({"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
                         "e1_masks": e1_mask, "e2_masks": e2_mask, "labels": tag2idx[item['relation']]})

    return data_set


def save_target(target, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(target))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        target = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(target))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        target = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(target))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class SentenceREDataset(Dataset):
    def __init__(self, data_file_path, target_path, pretrained_model_path=None):
        self.data_file_path = data_file_path
        self.pretrained_model_path = pretrained_model_path
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
        self.data_set = read_data(data_file_path, target_path, tokenizer=self.tokenizer)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]

    def collate_fn(self, batch_data):
        input_ids_list, token_type_ids_list, attention_mask_list, e1_masks_list, e2_masks_list, labels_list = [], [], [], [], [], []

        for instance in batch_data:
            # 按照batch中的最大数据长度,对数据进行padding填充
            input_ids_temp = instance["input_ids"]
            token_type_ids_temp = instance["token_type_ids"]
            attention_mask_temp = instance["attention_mask"]
            e1_masks_temp = instance["e1_masks"]
            e2_masks_temp = instance["e2_masks"]
            labels_temp = instance["labels"]

            # 添加到对应的list中
            input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
            token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
            e1_masks_list.append(torch.tensor(e1_masks_temp, dtype=torch.long))
            e2_masks_list.append(torch.tensor(e2_masks_temp, dtype=torch.long))
            labels_list.append(labels_temp)

        # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
        return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
                "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
                "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
                "e1_masks": pad_sequence(e1_masks_list, batch_first=True, padding_value=0),
                "e2_masks": pad_sequence(e2_masks_list, batch_first=True, padding_value=0),
                "labels": torch.tensor(labels_list, dtype=torch.long)}


if __name__ == "__main__":
    dev_dataset = SentenceREDataset(hparams.test_file, target_path=hparams.target_file,
                                    pretrained_model_path=hparams.bert_path)
    dev_loader = DataLoader(dev_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=dev_dataset.collate_fn)
    for i in dev_loader:
        print(i)
        break


