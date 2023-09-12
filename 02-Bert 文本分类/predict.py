# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time


def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x


def text_class_name(pred):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    print(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")
    
    
if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(device, args.save_model_best)

    texts = ["我们一起去打篮球吧！", "我喜欢踢足球！", "沈腾和马丽的新电影《独行月球》很好看", "昨天玩游戏，完了一整天",
             "现在的高考都已经开始分科考试了。", "中方：佩洛西如赴台将致严重后果", "现在的股票基金趋势很不好"]
    print("模型预测结果：")
    for text in texts:
        x = process_text(text, args.bert_pred)
        with torch.no_grad():
            pred = model(x)
        text_class_name(pred)
    end = time.time()
    print(f"耗时为：{end - start} s")
