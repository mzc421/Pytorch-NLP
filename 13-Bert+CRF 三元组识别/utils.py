# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import json
import torch
import torch.utils.data as tud
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from config import parsers


# 实体标注
START_TAG, STOP_TAG = "<START>", "<STOP>"
tag2idx = {START_TAG: 0, "O": 1, "B-SUB": 2, "I-SUB": 3, "B-OBJ": 4, "I-OBJ": 5, "B-BOTH": 6, "I-BOTH": 7, STOP_TAG: 8}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# 关系
relation = [line.strip() for line in open(parsers().relation, encoding="utf-8").readlines()]
relation2idx = {key: value for key, value in zip(relation, range(len(relation)))}
idx2relation = {idx: relation for relation, idx in relation2idx.items()}


class NERDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, tag2idx, relation2idx):
        super(NERDataset, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.data_set = []

        with open(data_path, encoding='utf-8') as f:
            all_data = f.readlines()

        for line in all_data:
            data = json.loads(line)
            # 得到文本
            text = data["text"]
            # 分词
            chars = [char for char in text]
            # 文本编码
            input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
            # 文本长度
            real_len = len(input_ids)

            # 整个句子标注 得到 tag_ids
            tags = ['O'] * real_len
            # mask 掩码
            sub_mask = [0] * real_len
            obj_mask = [0] * real_len

            # 加上起始符号和终止符号
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
            # 找出两个实体并标注
            subjects, objects = set(), set()

            sub = data["h"]
            subjects.add(sub["name"])
            sub_start, sub_end = sub["pos"]
            tags[sub_start] = 'B-SUB'
            tags[sub_start + 1:sub_end] = ['I-SUB'] * (sub_end - sub_start - 1)

            obj = data["t"]
            subjects.add(obj["name"])
            obj_start, obj_end = obj["pos"]
            tags[obj_start] = 'B-OBJ'
            tags[obj_start + 1:obj_end] = ['I-OBJ'] * (obj_end - obj_start - 1)

            # 合并 标注
            tags = ['O'] + tags + ['O']
            # 转换为数字
            tag_ids = [tag2idx[tag] for tag in tags]

            # 得到 subject mask 和 object mask
            sub_name = data["t"]["name"]
            for con in sub_name:
                index = chars.index(con)
                sub_mask[index] = 1
            sub_mask = [0] + sub_mask + [0]

            obj_name = data["h"]["name"]
            for con in obj_name:
                index = chars.index(con)
                obj_mask[index] = 1
            obj_mask = [0] + obj_mask + [0]

            # 关系
            label = relation2idx[data["relation"]]

            assert len(input_ids) == len(tag_ids) == len(sub_mask) == len(obj_mask)
            self.data_set.append(
                {"input_ids": input_ids, "tag_ids": tag_ids, "attention_mask": [1] * len(input_ids),
                 "sub_mask": sub_mask, "obj_mask": obj_mask, "label": label, "real_len": real_len + 2})

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]

    def collate_fn(self, batch_data):
        """
        DataLoader所需的collate_fun函数，将数据处理成tensor形式
        Args:
            batch_data: batch数据
        Returns:
        """
        input_ids_list, tag_ids_list, attention_mask_list, real_lens_list, sub_mask_list, obj_mask_list, labels_list, real_len_list = [], [], [], [], [], [], [], []

        for instance in batch_data:
            # 按照batch中的最大数据长度,对数据进行padding填充
            input_ids_temp = instance["input_ids"]
            tag_ids_temp = instance["tag_ids"]
            attention_mask_temp = instance["attention_mask"]
            sub_mask_temp = instance["sub_mask"]
            obj_mask_temp = instance["obj_mask"]
            label_temp = instance["label"]
            real_len_temp = instance["real_len"]

            # 添加到对应的list中
            input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
            tag_ids_list.append(torch.tensor(tag_ids_temp, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
            sub_mask_list.append(torch.tensor(sub_mask_temp, dtype=torch.long))
            obj_mask_list.append(torch.tensor(obj_mask_temp, dtype=torch.long))
            labels_list.append(label_temp)
            real_lens_list.append(real_len_temp)

        # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
        return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
                "tag_ids": pad_sequence(tag_ids_list, batch_first=True, padding_value=1),  # "O"对应的ID为1
                "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
                "sub_mask": pad_sequence(sub_mask_list, batch_first=True, padding_value=0),
                "obj_mask": pad_sequence(obj_mask_list, batch_first=True, padding_value=0),
                "labels": torch.tensor(labels_list, dtype=torch.long),
                "real_lens": real_lens_list}


if __name__ == "__main__":
    opt = parsers()

    dev_dataset = NERDataset(opt.dev_file, opt.bert_file, tag2idx, relation2idx)
    dev_dataloader = tud.DataLoader(dev_dataset, opt.batch_size, shuffle=True, collate_fn=dev_dataset.collate_fn)
    for i in dev_dataloader:
        print(i)
        break
