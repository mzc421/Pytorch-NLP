# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from model import BertNerModel
from transformers import BertTokenizer
import pickle as pkl
import torch
from config import parsers
import time


def load_model(model_path, class_num):
    global device
    model = BertNerModel(class_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def text_class_name(texts, pred, index_label):
    result = torch.argmax(pred[0], dim=1)
    result = result.cpu().numpy().tolist()

    pred_label = [index_label[i] for i in result]
    print("模型预测结果：")
    print(f"文本：{texts}\t预测的类别为：{pred_label[:len(texts)]}")


def pred_one():
    global args

    text = "李某某，男，2012年4月出生，本科学历，工科学士，毕业于电子科技大学。"
    dataset = pkl.load(open(args.data_pkl, "rb"))
    label_index, index_label = dataset[0], dataset[1]

    tokenizer = BertTokenizer.from_pretrained(args.bert_pred)

    text_id = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_len + 2,
                               padding="max_length", truncation=True, return_tensors="pt")

    model = load_model(args.save_model_best, len(label_index))

    text_id = text_id.to(device)
    with torch.no_grad():
        pred = model(text_id)
    text_class_name(text, pred, index_label)


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pred_one()  # 预测一条文本
    end = time.time()
    print(f"耗时为：{end - start} s")
