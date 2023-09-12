# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from config import parsers
# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。
# 所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码。
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


def read_data(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签、句子的最大长度
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)

    return texts, labels


class MyDataset(Dataset):
    def __init__(self, texts, labels=None, with_labels=True):
        self.all_text = texts
        self.all_label = labels
        self.max_len = parsers().max_len
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)

    def __getitem__(self, index):
        text = self.all_text[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.max_len,
                                      return_tensors='pt')  # Return torch.Tensor objects
        # shape [max_len]
        # tensor of token ids  torch.Size([max_len])
        token_ids = encoded_pair['input_ids'].squeeze(0)
        # binary tensor with "0" for padded values and "1"  for the other values  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:  # True if the dataset has labels
            label = int(self.all_label[index])
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label = read_data("./data/train.txt")
    print(train_text[0], train_label[0])
    trainDataset = MyDataset(train_text, labels=train_label, with_labels=True)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    for i, batch in enumerate(trainDataloader):
        print(batch[0], batch[1], batch[2], batch[3])
