# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import re
import json
from tqdm import tqdm
from config import parsers


def convert_data(line):
    head_name, tail_name, relation, text = re.split(r'\t', line)
    match_obj1 = re.search(head_name, text)
    match_obj2 = re.search(tail_name, text)
    if match_obj1 and match_obj2:  # 姑且使用第一个匹配的实体的位置
        head_pos = match_obj1.span()
        tail_pos = match_obj2.span()
        item = {
            'h': {
                'name': head_name,
                'pos': head_pos
            },
            't': {
                'name': tail_name,
                'pos': tail_pos
            },
            'relation': relation,
            'text': text
        }
        return item
    else:
        return None


def save_data(lines, file):
    with open(file, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, total=len(lines), desc=file):
            item = convert_data(line)
            if item is None:
                continue
            json_str = json.dumps(item, ensure_ascii=False)
            f.write('{}\n'.format(json_str))


def split_data(file):
    file_dir = os.path.dirname(file)
    train_file = os.path.join(file_dir, 'train.json')
    val_file = os.path.join(file_dir, 'val.json')
    test_file = os.path.join(file_dir, 'test.json')
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    train_lines = lines[:len(lines) * 6 // 10]
    val_lines = lines[len(lines) * 6 // 10:len(lines) * 8 // 10]
    test_lines = lines[len(lines) * 8 // 10:]

    train_lines = sorted(train_lines, key=lambda x: len(x.split("\t")[-1]))
    val_lines = sorted(val_lines, key=lambda x: len(x.split("\t")[-1]))
    test_lines = sorted(test_lines, key=lambda x: len(x.split("\t")[-1]))

    save_data(train_lines, train_file)
    save_data(val_lines, val_file)
    save_data(test_lines, test_file)


if __name__ == '__main__':
    split_data(parsers().all_data)

