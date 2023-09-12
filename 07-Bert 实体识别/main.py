# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai


import time
from tqdm import tqdm
from config import parsers
from utils import read_data, MyDataset, build_label_index
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import BertNerModel
from seqeval.metrics import f1_score, precision_score, recall_score


def prepare_data():
    global args
    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    test_text, test_label = read_data(args.test_file)
    label_to_index, index_to_label = build_label_index(train_label)

    trainDataset = MyDataset(train_text, label_to_index, labels=train_label, with_labels=True)
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    devDataset = MyDataset(dev_text, label_to_index, labels=dev_label, with_labels=True)
    devLoader = DataLoader(devDataset, batch_size=args.batch_size, shuffle=False)

    testDataset = MyDataset(test_text, label_to_index, labels=test_label, with_labels=True)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    return trainLoader, devLoader, testLoader, label_to_index, index_to_label


if __name__ == "__main__":
    start = time.time()
    args = parsers()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    train_loader, dev_loader, test_loader, label_index, index_label = prepare_data()

    model = BertNerModel(len(label_index)).to(device)
    opt = torch.optim.AdamW(model.parameters(), args.learn_rate)
    loss_fun = nn.CrossEntropyLoss()

    f1_max = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum, num = 0, 0
        pbar = tqdm(train_loader)
        for batch_text, batch_label in pbar:
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)

            pred = model(batch_text)

            loss = loss_fun(pred.reshape(-1, pred.shape[-1]), batch_label.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_sum += loss
            num += 1

            pbar.set_description('epoch: {}/{}'.format(epoch + 1, args.epochs))  # set_description()设置进度条前方信息
            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})  # set_postfix()设置进度条后方信息

        loss_avg = loss_sum / num
        print(f"train epoch:{epoch+1}\tloss:{loss_avg:.2f}")

        model.eval()

        all_pre = []
        all_tag = []
        for batch_text, batch_label in dev_loader:
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            pred = model(batch_text)

            pred_label = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
            tag_label = batch_label.cpu().numpy().tolist()

            for pred, tag in zip(pred_label, tag_label):
                p = [index_label[i] for i in pred]
                t = [index_label[i] for i in tag]

                all_pre.append(p)
                all_tag.append(t)

        f1 = f1_score(all_tag, all_pre)
        precision = precision_score(all_tag, all_pre)
        recall = recall_score(all_tag, all_pre)

        print(f"dev f1:{f1}, precision:{precision}，recall:{recall}")
        if f1_max < f1:
            f1_max = f1
            torch.save(model.state_dict(), args.save_model_best)

    model.eval()
    torch.save(model.state_dict(), args.save_model_last)

    all_pre = []
    all_tag = []
    for batch_text, batch_label in test_loader:
        batch_text = batch_text.to(device)
        batch_label = batch_label.to(device)
        pred = model(batch_text)

        pred_label = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
        tag_label = batch_label.cpu().numpy().tolist()

        for pred, tag in zip(pred_label, tag_label):
            p = [index_label[i] for i in pred]
            t = [index_label[i] for i in tag]

            all_pre.append(p)
            all_tag.append(t)

    f1 = f1_score(all_tag, all_pre)
    precision = precision_score(all_tag, all_pre)
    recall = recall_score(all_tag, all_pre)

    print(f"test f1:{f1}, precision:{precision}，recall:{recall}")

    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f} min")

