# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import re
import torch
from utils import MyTokenizer, get_idx2tag, convert_pos_to_mask
from model import SentenceRE
from config import hparams


def process_data(tokenizer, text, entity1, entity2, device):
    # 找到两个实体在句子中的位置
    match_obj1 = re.search(entity1, text)
    match_obj2 = re.search(entity2, text)

    # 得到所应值
    e1_pos = match_obj1.span()
    e2_pos = match_obj2.span()

    # 构建格式
    item = {
        'h': {
            'name': entity1,
            'pos': e1_pos
        },
        't': {
            'name': entity2,
            'pos': e2_pos
        },
        'text': text
    }

    # 编码格式
    tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
    encoded = tokenizer.bert_tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')

    input_ids = encoded['input_ids'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])]).to(device)
    e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])]).to(device)
    return input_ids, token_type_ids, attention_mask, e1_mask, e2_mask


def predict(hparams):
    device = hparams.device
    target_file = hparams.target_file

    bert_path = hparams.bert_path
    model_best_bin = hparams.model_best_bin

    idx2tag = get_idx2tag(target_file)
    model = SentenceRE(hparams).to(device)
    model.load_state_dict(torch.load(model_best_bin), strict=False)
    model.eval()
    tokenizer = MyTokenizer(bert_path)

    text = input("输入中文句子：")
    entity1 = input("句子中的实体1：")
    entity2 = input("句子中的实体2：")
    input_ids, token_type_ids, attention_mask, e1_mask, e2_mask = process_data(tokenizer, text, entity1, entity2, device)

    with torch.no_grad():
        preds = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
    preds = preds.to(torch.device('cpu'))
    print("在【{}】中【{}】与【{}】的关系为：{}".format(text, entity1, entity2, idx2tag[preds.argmax(0).item()]))


predict(hparams)

