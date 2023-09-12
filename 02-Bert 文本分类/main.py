# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from model import MyModel
from torch.optim import AdamW
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time
from test import test_data


if __name__ == "__main__":
    start = time.time()
    args = parsers()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_text, train_label, max_len = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    args.max_len = max_len

    train_dataset = MyDataset(train_text, train_label, args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = MyDataset(dev_text, dev_label, args.max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    opt = AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = nn.CrossEntropyLoss()

    acc_max = float("-inf")
    for epoch in range(args.epochs):
        loss_sum, count = 0, 0
        model.train()
        for batch_index, (batch_text, batch_label) in enumerate(train_dataloader):
            batch_label = batch_label.to(device)
            pred = model(batch_text)

            loss = loss_fn(pred, batch_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
            count += 1

            # 打印内容
            if len(train_dataloader) - batch_index <= len(train_dataloader) % 1000 and count == len(train_dataloader) % 1000:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

            if batch_index % 1000 == 999:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch_text, batch_label in dev_dataloader:
                batch_label = batch_label.to(device)
                pred = model(batch_text)

                pred = torch.argmax(pred, dim=1).cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()

                all_pred.extend(pred)
                all_true.extend(label)

        acc = accuracy_score(all_pred, all_true)
        print(f"dev acc:{acc:.4f}")
        if acc > acc_max:
            print(acc, acc_max)
            acc_max = acc
            torch.save(model.state_dict(), args.save_model_best)
            print(f"以保存最佳模型")

    torch.save(model.state_dict(), args.save_model_last)

    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
    test_data()
