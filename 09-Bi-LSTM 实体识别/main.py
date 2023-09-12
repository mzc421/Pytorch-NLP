# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import torch
from seqeval.metrics import f1_score
from config import parsers
from utils import prepare_data
from model import BiLSTMModel


if __name__ == "__main__":
    args = parsers()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader, dev_loader, test_loader, index_label, corpus_num, class_num = prepare_data()

    model = BiLSTMModel(corpus_num, class_num, args.embedding_num, args.hidden_num, args.bi).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    f1_max = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0, 0
        for batch_data, batch_label in train_loader:
            loss = model(batch_data, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_sum += loss
            count += 1

        print(f"epoch:{epoch+1} \t train loss:{loss_sum/count:.3f}")

        model.eval()
        all_pred, all_label = [], []
        for batch_data, batch_label in dev_loader:
            pred = model(batch_data)
            all_pred.append([index_label[i] for i in pred.cpu().numpy().tolist()])
            all_label.append([index_label[i] for i in batch_label.cpu().numpy().reshape(-1).tolist()])
        f1 = f1_score(all_label, all_pred)
        print(f"epoch:{epoch + 1}\tf1:{f1:.3f}")

        if f1_max < f1:
            f1_max = f1
            torch.save(model.state_dict(), args.save_model_best)
            print("保存最佳模型")
    # 在导出模型之前，请先调用 model.eval() 或 model.train(False)，
    # 以将模型转换为推理模式，这一点很重要。
    # 这是必需的，因为像 dropout 或 batchnorm 这样的运算符在推断和训练模式下的行为会有所不同。
    model.eval()
    torch.save(model.state_dict(), args.save_model_last)

    all_pred, all_label = [], []
    for batch_data, batch_label in test_loader:
        pred = model(batch_data)
        all_pred.append([index_label[i] for i in pred.cpu().numpy().tolist()])
        all_label.append([index_label[i] for i in batch_label.cpu().numpy().reshape(-1).tolist()])
    f1 = f1_score(all_label, all_pred)
    print(f"test f1:{f1:.3f}")
