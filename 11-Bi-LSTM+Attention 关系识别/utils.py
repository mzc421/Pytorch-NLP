# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
from tqdm import tqdm
from config import hparams as hp
from torch.utils.data import Dataset, DataLoader
import pickle

# 关系字典
relation2id = {}
with open(hp.target_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        relation2id[line.split()[0]] = int(line.split()[1])
id2relation = {value: key for key, value in relation2id.items()}

with open(hp.all_data, 'r', encoding='utf-8') as f:
    all_text = f.readlines()

max_text = max([len(i) for i in all_text])
texts_one = list(set(one for data in all_text for one in data))
word2id = {text: index for index, text in enumerate(texts_one)}
word2id["BLANK"] = len(word2id) + 1
word2id["UNKNOW"] = len(word2id) + 1

param_dict = {"word2id": word2id, 'word2id_len': len(word2id), 'max_text': max_text + 2}
with open(hp.param_dict, 'wb') as f:
    pickle.dump(param_dict, f)


def pos(num):
    """
    实体相对位置
    """
    if num < -(max_text//2):
        return 0
    elif -(max_text//2) <= num <= (max_text//2):
        return num + (max_text//2)
    if num > (max_text//2):
        return max_text + 1


class NerDataLoader(Dataset):
    def __init__(self, mode="Train"):
        self.mode = mode

        # -----------------------------------确保每个批次都是偶数-----------------------------------------
        # 计算80%和20%的切分点
        num_rate = int(0.8 * len(all_text))
        if self.mode == "Train":
            temp_data = all_text[:num_rate]
            div_num = len(temp_data) - len(temp_data) // hp.batch_size * 2
            if div_num != 0:
                need_num = hp.batch_size - div_num
                all_data = all_text[:num_rate + need_num]
            else:
                all_data = temp_data
        else:
            temp_data = all_text[num_rate:]
            div_num = len(temp_data) - len(temp_data) // hp.batch_size * 2
            if div_num != 0:
                need_num = hp.batch_size - div_num
                all_data = all_text[num_rate - need_num:]
            else:
                all_data = temp_data
        # ---------------------------------------------------------------------------------------

        self.entity1, self.entity2, relations, texts = [], [], [], []
        for lines in tqdm(all_data, ncols=80, desc=f"数据处理 {self.mode}"):
            entity1, entity2, relation, text = lines.split("\t")
            temp_entity1, temp_entity2 = [], []
            # 得到实体的位置
            index1 = text.index(entity1)
            index2 = text.index(entity2)

            # 以实体为中心，得到实体列表
            for i in range(len(text)):
                temp_entity1.append(i - index1)
                temp_entity2.append(i - index2)

            texts.append([text[:-1]])
            self.entity1.append(temp_entity1)
            self.entity2.append(temp_entity2)
            relations.append([relation])

        self.text = [[word2id[word] for word in text[0]] for text in texts]
        self.relation = [[relation2id[word] for word in relation] for relation in relations]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.entity1[item], self.entity2[item], self.relation[item], self.text[item]

    def padding(self, batch_data):
        """补全位 max_len 长度"""
        # 找到同一个批次的最大长度
        max_len = max([len(i[0]) for i in batch_data])

        text_position1 = [[pos(num) for num in i[0]] + [81] * (max_len - len(i[0])) for i in batch_data]
        text_position2 = [[pos(num) for num in i[1]] + [81] * (max_len - len(i[1])) for i in batch_data]
        text_label = [i[2][0] for i in batch_data]
        text = [i[3] + [word2id["BLANK"]] * (max_len - len(i[3])) for i in batch_data]

        return (torch.tensor(text, dtype=torch.long), torch.tensor(text_position1, dtype=torch.long),
                torch.tensor(text_position2, dtype=torch.long), torch.tensor(text_label, dtype=torch.long))


if __name__ == "__main__":
    datasets = NerDataLoader("Test")
    dataloader = DataLoader(datasets, hp.batch_size, shuffle=False, num_workers=2, collate_fn=datasets.padding)

    for batch_data in dataloader:
        sentence = batch_data[0].to(hp.device)
        pos1 = batch_data[1].to(hp.device)
        pos2 = batch_data[2].to(hp.device)
        tag = batch_data[3].to(hp.device)
        # break
