# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os.path


"""
BIO：B-begin,I-inside,O-outside。
B-X代表实体X的开头、I-X代表实体的结尾、O代表不属于任何类型的
BMOES：B-begin,M-middle,E-end,S-single。
B-X表示一个词的词首位值，M-X示一个词的中间位置，E-X表示一个词的末尾位置，S-X表示一个单独的字词
BIOES：B-begin,I-inside,O-outside,E-end,S-single。
B-x表示开始，I-x表示内部，O表示非实体，E-x实体尾部，S-x表示该词本身就是一个实体
"""

tagging_method = "BIOES"


def BIO(tag_list, index_start_tag, i, feature_tag_dict, keyword):
    """
    BIO 标注
    """
    # B
    if index_start_tag == i:
        if tag_list[i] == 'O':
            tag_list[i] = "B-" + feature_tag_dict[keyword]

    # I
    else:
        if tag_list[i] == 'O':
            tag_list[i] = "I-" + feature_tag_dict[keyword]


def BMES(tag_list_BMES, index_start_tag, i, keyword):
    """
    BMES 标注  扁豆细菌性疫病的危害作物是扁豆吗？
    """
    # B
    if index_start_tag == i:
        if tag_list_BMES[i] == 'S':
            tag_list_BMES[i] = "B"

    # M
    elif i != index_start_tag + len(keyword) - 1 and len(keyword) != 1:
        if tag_list_BMES[i] == 'S':
            tag_list_BMES[i] = "M"

    # E
    elif i == index_start_tag + len(keyword) - 1 and len(keyword) != 1:
        if tag_list_BMES[i] == 'S':
            tag_list_BMES[i] = "E"


def BIOES(tag_list, index_start_tag, i, feature_tag_dict, keyword):
    """
    BMES 标注  扁豆细菌性疫病的危害作物是扁豆吗？
    """
    # B
    if index_start_tag == i and len(keyword) != 1:
        if tag_list[i] == 'O':
            tag_list[i] = "B-" + feature_tag_dict[keyword]

    # I
    elif len(keyword) != 1 and i != index_start_tag + len(keyword) - 1:
        if tag_list[i] == 'O':
            tag_list[i] = "I-" + feature_tag_dict[keyword]

    # E
    elif len(keyword) != 1 and i == index_start_tag + len(keyword) - 1:
        if tag_list[i] == 'O':
            tag_list[i] = "E-" + feature_tag_dict[keyword]

    # S
    elif len(keyword) == 1:
        if tag_list[i] == 'O':
            tag_list[i] = "S-" + feature_tag_dict[keyword]


def save_tagging(tag_list, file_output, line):
    print(tag_list)
    with open(file_output, 'a', encoding='utf-8') as output_f:
        for w, t in zip(line.strip(), tag_list):
            output_f.write(w + " " + t + '\n')
        output_f.write('\n')
    print("-"*100)


def get_dict():
    """
    创建 feature_label_dict,以特征词作为 key, tag 作为 value
    """
    feature_label = open("./data/word_dict.txt", "r", encoding="utf-8").read().split("\n")
    feature_label_dict = {}
    for line in feature_label:
        try:
            feature, label = line.split(" ")
            feature_label_dict[feature] = label
        except:
            with open('./data/error.txt', 'a', encoding='utf-8') as f:
                f.write(line + "\n")

    # print(feature_label_dict)
    return feature_label_dict


def main():
    """
    进行实体自动标注，用字典中的 key 作为关键词去匹配未标注的文本，将匹配的内容进行标注未 value
    """
    feature_tag_dict = get_dict()
    file_input = './data/noTagging.txt'
    file_output = './data/Tagging.txt'
    index_log = 0
    if os.path.exists(file_output):
        os.remove(file_output)
    with open(file_input, 'r', encoding='utf-8') as f_input:
        for line in f_input.readlines():
            print(line, end="")
            # O
            tag_list = ["O" for i in range(len(line.strip()))]
            tag_list_BMES = ["S" for i in range(len(line.strip()))]

            for keyword in feature_tag_dict.keys():
                while True:
                    index_start_tag = line.find(keyword, index_log)
                    # 当前关键词查找不到，跳出循环进入下一个关键词
                    if index_start_tag == -1:
                        index_log = 0
                        break
                    index_log = index_start_tag + 1
                    print(keyword, ":", index_start_tag)
                    # 只对未标注过的数据进行标注，防止出现嵌套标注
                    for i in range(index_start_tag, index_start_tag + len(keyword)):
                        # BIO 标注
                        if tagging_method == "BIO":
                            BIO(tag_list, index_start_tag, i, feature_tag_dict, keyword)
                        elif tagging_method == "BMES":
                            BMES(tag_list_BMES, index_start_tag, i, keyword)
                        else:
                            BIOES(tag_list, index_start_tag, i, feature_tag_dict, keyword)

            if tagging_method == "BMES":
                save_tagging(tag_list_BMES, file_output, line)
            else:
                save_tagging(tag_list, file_output, line)


main()
