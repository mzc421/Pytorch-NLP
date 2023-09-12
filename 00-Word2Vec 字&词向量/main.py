# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
from gensim.models import Word2Vec
import gensim
from gensim.models import KeyedVectors
import jieba
import pickle as pkl


def train_word():
    print("*"*5 + "训练" + "*"*5)
    if os.path.exists('word_data.vector') and os.path.exists('word.model') and os.path.exists("WordPartialWeight.pkl"):
        return
    # 得到每一行的数据 []
    datas = open('data/test.txt', 'r', encoding='utf-8').read().split("\n")
    # 得到一行的单个字 [[],...,[]]
    word_datas = [[i for i in data[:-2] if i != " "] for data in datas]

    model = Word2Vec(
        word_datas,  # 需要训练的文本
        vector_size=10,   # 词向量的维度
        window=2,  # 句子中当前单词和预测单词之间的最大距离
        min_count=1,  # 忽略总频率低于此的所有单词 出现的频率小于 min_count 不用作词向量
        workers=8,  # 使用这些工作线程来训练模型（使用多核机器进行更快的训练）
        sg=0,  # 训练方法 1：skip-gram 0；CBOW。
        epochs=10  # 语料库上的迭代次数
    )

    # 字向量保存
    model.wv.save_word2vec_format('word_data.vector',   # 保存路径
                                  binary=False  # 如果为 True，则数据将以二进制 word2vec 格式保存，否则将以纯文本格式保存
                                  )

    # 模型保存
    model.save('word.model')

    # 模型中的 wv 和 syn1neg 都可以单独保存
    pkl.dump([model.wv.index_to_key, model.wv.key_to_index, model.wv.vectors], open("WordPartialWeight.pkl", "wb"))


def use_word():
    print("*"*5 + "使用" + "*"*5)
    # 1 通过模型加载词向量(recommend)
    model = gensim.models.Word2Vec.load('word.model')
    # 2 通过字向量加载
    vector = KeyedVectors.load_word2vec_format('word_data.vector')

    lis = model.wv.index_to_key
    # print(lis)
    # print(len(lis))
    print("通过模型查看：", model.wv['提'])
    print("通过字向量查看：", vector['提'])

    print(vector.most_similar('提',
                              topn=3  # 返回前 3 个相似键的数量
                              ))


def train_words():
    print("*"*5 + "训练" + "*"*5)
    if os.path.exists('words_data.vector') and os.path.exists('words.model') and os.path.exists("WordsPartialWeight.pkl"):
        return
    datas = open("data/test.txt", "r", encoding="utf-8").read().split("\n")
    words_datas = [[i for i in (jieba.cut(data)) if i != " "] for data in datas]
    model = Word2Vec(words_datas, vector_size=10, window=2, min_count=1, workers=8,  sg=0, epochs=10)

    model.wv.save_word2vec_format('words_data.vector', binary=False)

    model.save('words.model')
    pkl.dump([model.wv.index_to_key, model.wv.key_to_index, model.wv.vectors], open("WordsPartialWeight.pkl", "wb"))


def use_words():
    print("*"*5 + "使用" + "*"*5)
    # 1 通过模型加载词向量(recommend)
    model = gensim.models.Word2Vec.load('words.model')
    # 2 通过词向量加载
    vector = KeyedVectors.load_word2vec_format('words_data.vector')

    dic = model.wv.key_to_index
    # print(dic)
    # print(len(dic))

    print("通过模型进行查看：", model.wv['提升'])
    print("通过字向量进行查看：", vector['提升'])
    print(vector.most_similar('提升', topn=3))


def word():
    print("*"*10 + "字" + "*"*10)
    train_word()
    use_word()


def words():
    print("*"*10 + "词" + "*"*10)
    train_words()
    use_words()


def read_pkl():
    dataset = pkl.load(open("WordPartialWeight.pkl", "rb"))
    index_to_key, key_to_index,  vector = dataset[0], dataset[1], dataset[2]
    # print("index_to_key:", index_to_key)
    # print("key_to_index:", key_to_index)
    # print("vector:", vector)


if __name__ == "__main__":
    # 字
    word()
    # 词
    words()
    read_pkl()

