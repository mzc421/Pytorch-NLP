# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import numpy as np
import torch
from model import BiLSTM_ATT
from config import hparams as hp
from utils import pos, id2relation
import pickle

with open(hp.param_dict, 'rb') as f:
    my_dict = pickle.load(f)

word2id = my_dict["word2id"]
word2id_len = my_dict["word2id_len"]
max_text = my_dict["max_text"]


def load():
    hp.embedding_size = word2id_len
    hp.pos_size = max_text
    hp.batch_size = 1
    model = BiLSTM_ATT(hp).to(hp.device)
    model.load_state_dict(torch.load(hp.model_best))
    model.eval()
    return model


def process_text(content, ent1, ent2):
    entity1, entity2 = [], []
    # 得到实体的位置
    index1 = content.index(ent1)
    index2 = content.index(ent2)

    # 以实体为中心，得到实体列表
    for i in range(len(content)):
        entity1.append(i - index1)
        entity2.append(i - index2)

    text = [[word2id.get(word, word2id.get("UNKNOW")) for word in content]]
    text_position1 = [[pos(num) for num in entity1]]
    text_position2 = [[pos(num) for num in entity2]]

    return (torch.tensor(text, dtype=torch.long), torch.tensor(text_position1, dtype=torch.long),
            torch.tensor(text_position2, dtype=torch.long))


if __name__ == "__main__":
    model = load()
    text = input("请输入文本：")
    ent1 = input("请输入文本中的实体1：")
    ent2 = input("请输入文本中的实体2：")
    loader = process_text(text, ent1, ent2)
    with torch.no_grad():
        sentence = loader[0].to(hp.device)
        pos1 = loader[1].to(hp.device)
        pos2 = loader[2].to(hp.device)

        y = model(sentence, pos1, pos2)
        y = np.argmax(y.cpu().data.numpy(), axis=1)[0]
        print(f"{text}中的{ent1}与{ent2}的关系为：{id2relation[y]}")

