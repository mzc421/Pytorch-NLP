# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from model import TextCNNModel
import torch
from torch.utils.data import DataLoader
from utils import TextDataset
from config import parsers
import pickle as pkl
import time


def load_model(embedding, max_len, class_num, num_filters, model_path, device):
    """加载模型"""
    model = TextCNNModel(embedding, max_len, class_num, num_filters).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def process_text(text, word_2_index, max_len):
    """数据处理"""
    text_dataset = TextDataset([text], [0], word_2_index, max_len)
    test_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False)
    for batch_text, batch_label in test_dataloader:
        return batch_text


def text_class_name(pred):
    """分类"""
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    print(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = pkl.load(open(args.data_pkl, "rb"))
    word_2_index, words_embedding = dataset[0], dataset[1]

    # 加载模型
    model = load_model(words_embedding, args.max_len, args.class_num, args.num_filters, args.save_model_best, device)

    texts = ["我们一起去打篮球吧！", "沈腾和马丽的新电影《独行月球》很好看", "昨天玩游戏，完了一整天",
             "现在的高考都已经开始分科考试了。", "中方：佩洛西如赴台将致严重后果", "现在的股票基金趋势很不好"]
    print("模型预测结果：")
    for text in texts:
        text_id = process_text(text, word_2_index, args.max_len)
        with torch.no_grad():
            # 进行预测
            model.eval()
            text_id = text_id.to(device)
            pred = model(text_id)
            text_class_name(pred)
    end = time.time()
    print(f"耗时为：{end - start} s")
