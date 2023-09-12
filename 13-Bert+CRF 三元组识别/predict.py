# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
from transformers import BertTokenizer
from model import BertForRE
import time
from config import parsers


if __name__ == '__main__':
    opt = parsers()

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = BertForRE.from_pretrained(opt.save_pretrained_best)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(opt.bert_file)
    startTime = time.time()

    # 预测文本
    text = "葛淑珍离开赵本山18年，走出自己的路。"
    # 分词
    chars = [char for char in text]

    chars = ['[CLS]'] + chars + ['[SEP]']
    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
    attention_mask = [1] * len(input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        model.predict(text, chars, input_ids, attention_mask)

    endTime = time.time()
    print(f"运行时间：{endTime-startTime}s")
