# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM_ATT
from config import hparams as hp
from utils import word2id, max_text, relation2id, NerDataLoader
from torch.utils.data import DataLoader


def train(train_loader, test_loader, model, criterion):
    f1_max = 0
    for epoch in range(hp.epochs):
        flag = False
        epoch_train_acc, epoch_train_total, epoch_train_loss = 0, 0, 0
        # 更新进度条的前缀文本
        for batch_data in train_loader:
            sentence = batch_data[0].to(hp.device)
            pos1 = batch_data[1].to(hp.device)
            pos2 = batch_data[2].to(hp.device)
            tag = batch_data[3].to(hp.device)

            # [batch_size, tag_size]
            y = model(sentence, pos1, pos2)
            loss = criterion(y, tag)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            # 计算训练过程中的准确率
            y = np.argmax(y.cpu().data.numpy(), axis=1)

            for y1, y2 in zip(y, tag):
                epoch_train_total += 1
                if y1 == y2:
                    epoch_train_acc += 1

        # 每一类的预测结果数量
        count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 每一类的真实结果数量
        count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 每一类的预测正确结果数量
        count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with torch.no_grad():
            for batch_data in test_loader:
                sentence = batch_data[0].to(hp.device)
                pos1 = batch_data[1].to(hp.device)
                pos2 = batch_data[2].to(hp.device)
                tag = batch_data[3].to(hp.device)

                y = model(sentence, pos1, pos2)
                y = np.argmax(y.cpu().data.numpy(), axis=1)

                for y1, y2 in zip(y, tag):
                    count_predict[y1] += 1
                    count_total[y2] += 1
                    if y1 == y2:
                        count_right[y1] += 1

        precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(count_predict)):
            if count_predict[i] != 0:
                precision[i] = float(count_right[i]) / count_predict[i]
            if count_total[i] != 0:
                recall[i] = float(count_right[i]) / count_total[i]

        precision = sum(precision) / len(relation2id)
        recall = sum(recall) / len(relation2id)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1_max < f1:
            f1_max = f1
            torch.save(model.state_dict(), hp.model_best)
            flag = True

        torch.save(model.state_dict(), hp.model_last)
        print(f"[{epoch}/{hp.epochs}] train: loss={100 * float(epoch_train_acc) / epoch_train_total: .4f}%;"
              f"test: acc={100 * precision: .4f}% recall={100 * recall: .4f}% F1={100 * f1: .4f}%"
              f" {'*' if flag else ''}")


if __name__ == "__main__":
    train_datasets = NerDataLoader("Train")
    train_loader = DataLoader(train_datasets, hp.batch_size, shuffle=False, num_workers=2,
                              collate_fn=train_datasets.padding)

    test_datasets = NerDataLoader("Test")
    test_loader = DataLoader(test_datasets, hp.batch_size, shuffle=False, num_workers=2,
                             collate_fn=test_datasets.padding)

    # 更新词嵌入的维度
    hp.embedding_size = len(word2id)
    hp.pos_size = max_text + 2
    # model
    model = BiLSTM_ATT(hp).to(hp.device)
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    train(train_loader, test_loader, model, criterion)


