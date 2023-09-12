# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from config import parsers
from utils import build_corpus, MyDataset
from model import BiLSTM_CRF
import pickle as pkl


def data_loader():
    args = parsers()

    # 构造一些训练数据,测试数据
    train_data, train_tag, train_word2index, train_tag2index = build_corpus('train', args.train_file)
    dev_data, dev_tag = build_corpus("dev", args.dev_file)
    test_data, test_tag = build_corpus("test", args.test_file)

    tag_to_index = train_tag2index
    START_TAG, STOP_TAG = "<START>", "<STOP>"
    tag_to_index.update({START_TAG: len(tag_to_index), STOP_TAG: len(tag_to_index) + 1})

    train_dataset = MyDataset(train_data, train_tag, train_word2index, tag_to_index)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, train_word2index, tag_to_index)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_test_batch_size, shuffle=False,
                            collate_fn=dev_dataset.pro_batch_data)

    test_dataset = MyDataset(test_data, test_tag, train_word2index, tag_to_index)
    test_loader = DataLoader(test_dataset, batch_size=args.dev_test_batch_size, shuffle=False,
                             collate_fn=test_dataset.pro_batch_data)

    return train_loader, dev_loader, test_loader, train_word2index, tag_to_index


def evaluate(dataloader):
    global model, device
    all_pre, all_tag = [], []
    with torch.no_grad():
        for sentences, tags in tqdm(dataloader):
            sentences, tags = sentences.to(device), tags.to(device)
            sentences.view(-1)
            tags.view(-1)

            pre_score, pre_tag = model.forward(sentences)
            # 预测的值放入集合中用于计算 f1_score, f1_score 的输入是两个 list 和一个 average 函数
            all_pre.extend(pre_tag)
            # 把测试集的 tag 拍平
            tags_flat = tags.detach().cpu().reshape(-1).tolist()
            # 标签值放入集合
            all_tag.extend(tags_flat)

    # 计算 f1_score
    score = f1_score(all_tag, all_pre, average="micro")
    return score


if __name__ == "__main__":
    args = parsers()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, dev_loader, test_loader, train_word2index, tag_to_index = data_loader()

    pkl.dump([train_word2index, tag_to_index], open(args.data_pkl, "wb"))

    # 定义模型
    model = BiLSTM_CRF(len(train_word2index), tag_to_index, args.embedding_dim, args.hidden_dim).to(device)
    # 优化函数
    optimizer = optim.AdamW(model.parameters(), lr=args.learn_rate, weight_decay=1e-4)

    score_max = float("-inf")
    for epoch in range(args.epochs):
        save_flag = False
        model.train()
        i = 0
        for sentences, tags in tqdm(train_loader):
            sentences, tags = sentences.to(device), tags.to(device)
            sentences = sentences.reshape(-1)
            tags = tags.reshape(-1)

            model.zero_grad()
            # 第二步，得到 loss
            loss = model.neg_log_likelihood(sentences, tags).cuda()
            # 第三步，计算 loss，梯度，通过 optimier 更新参数
            loss.backward()
            optimizer.step()
            i += 1
            if i == 10:
                break
        model.eval()
        score = evaluate(dev_loader)
        if score_max < score:
            save_flag = True
            torch.save(model.state_dict(), args.save_model_best)
        print(f"epoch:[{epoch+1}/{args.epochs}]\ttrain_loss:{loss.item():7.3f}\tdev f1_score:{score:.3f}\tsave model:{save_flag}")

    model.eval()
    torch.save(model.state_dict(), args.save_model_last)

    score = evaluate(test_loader)
    print(f"test f1_score:{score:.4f}")
