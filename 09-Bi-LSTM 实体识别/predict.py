# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from model import BiLSTMModel
import pickle as pkl
import torch
from config import parsers


def predict():
    global word_index, index_label, corpus_num, class_num, device, args
    model = BiLSTMModel(corpus_num, class_num, args.embedding_num, args.hidden_num, args.bi).to(device)
    model.load_state_dict(torch.load(args.save_model_best))
    model.eval()
    text = "李某某，男，2012年4月出生，本科学历，工科学士，毕业于电子科技大学。"
    text_id = [word_index.get(i, word_index["<UNK>"]) for i in text]
    text_id = torch.tensor(text_id, dtype=torch.int64, device=device)
    model(text_id)
    pred = [index_label[i] for i in model.pred.cpu().numpy().tolist()]
    result = []
    for w, s in zip(text, pred):
        result.append(f"{w}_{s}")
    print(f"预测结果：{result}")


if __name__ == "__main__":
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = pkl.load(open(args.data_pkl, "rb"))
    word_index, index_label, corpus_num, class_num = dataset[0], dataset[2], dataset[3], dataset[4]
    predict()

# 李川，男，2002年4月出生，本科学历，工科学士，毕业于电子科技大学成都学院。
