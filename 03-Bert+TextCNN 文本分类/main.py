# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import os
import time
from tqdm import tqdm
from config import parsers
from utils import read_data, MyDataset
from torch.utils.data import DataLoader
from model import BertTextModel_encode_layer, BertTextModel_last_layer
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score


def train(model, device, trainLoader, opt, epoch):
    model.train()
    loss_sum, count = 0, 0
    for batch_index, batch_con in enumerate(trainLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)

        opt.zero_grad()
        loss = loss_fn(pred, batch_con[-1])
        loss.backward()
        opt.step()
        loss_sum += loss
        count += 1

        if len(trainLoader) - batch_index <= len(trainLoader) % 1000 and count == len(trainLoader) % 1000:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0

        if batch_index % 1000 == 999:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0


def dev(model, device, devLoader, save_best):
    global acc_min
    model.eval()
    all_true, all_pred = [], []
    for batch_con in tqdm(devLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)

        pred = torch.argmax(pred, dim=1)

        pred_label = pred.cpu().numpy().tolist()
        true_label = batch_con[-1].cpu().numpy().tolist()

        all_true.extend(true_label)
        all_pred.extend(pred_label)

    acc = accuracy_score(all_true, all_pred)
    print(f"dev acc:{acc:.4f}")

    if acc > acc_min:
        acc_min = acc
        torch.save(model.state_dict(), save_best)
        print(f"以保存最佳模型")


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)

    trainData = MyDataset(train_text, train_label, with_labels=True)
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)

    devData = MyDataset(dev_text, dev_label, with_labels=True)
    devLoader = DataLoader(devData, batch_size=args.batch_size, shuffle=True)

    root, name = os.path.split(args.save_model_best)
    save_best = os.path.join(root, str(args.select_model_last) + "_" +name)
    root, name = os.path.split(args.save_model_last)
    save_last = os.path.join(root, str(args.select_model_last) + "_" +name)

    # 选择模型
    if args.select_model_last:
        # 模型1
        model = BertTextModel_last_layer().to(device)
    else:
        # 模型2
        model = BertTextModel_encode_layer().to(device)

    opt = AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = CrossEntropyLoss()

    acc_min = float("-inf")
    for epoch in range(args.epochs):
        train(model, device, trainLoader, opt, epoch)
        dev(model, device, devLoader, save_best)

    model.eval()
    torch.save(model.state_dict(), save_last)

    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
