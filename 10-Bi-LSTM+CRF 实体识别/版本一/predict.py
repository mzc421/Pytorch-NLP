# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import pickle as pkl
import torch
from config import parsers
from model import BiLSTM_CRF


def predict():
    global model, device, word_2_index, index_2_tag
    while True:
        text = input("请输入：")
        text_index = [[word_2_index.get(i, word_2_index["<UNK>"]) for i in text] + [word_2_index["<END>"]]]

        text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
        pre = model(text_index)
        pre = [index_2_tag[i] for i in pre[0]]
        print([f'{w}_{s}' for w, s in zip(text, pre)])


if __name__ == "__main__":
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dataset = pkl.load(open(args.data_pkl, "rb"))
    word_2_index, tag_2_index, index_2_tag = dataset[0], dataset[1], dataset[2]
    model = BiLSTM_CRF(len(word_2_index), tag_2_index, args.embedding_num, args.hidden_num, word_2_index["<PAD>"]).to(device)
    model.load_state_dict(torch.load(args.save_model_best))
    model.eval()

    predict()

# 李某某，男，2012年4月出生，本科学历，工科学士，毕业于电子科技大学。
