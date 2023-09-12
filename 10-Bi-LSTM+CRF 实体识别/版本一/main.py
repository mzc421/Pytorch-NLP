# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from torch.utils.data import DataLoader
import torch
from seqeval.metrics import f1_score
from config import parsers
from utils import build_corpus, MyDataset
from model import BiLSTM_CRF
import pickle as pkl


def test():
    global word_2_index, model, index_2_tag, device
    while True:
        text = input("请输入：")
        text_index = [[word_2_index.get(i, word_2_index["<UNK>"]) for i in text] + [word_2_index["<END>"]]]

        text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
        pre = model.test(text_index, [len(text) + 1])
        pre = [index_2_tag[i] for i in pre]
        print([f'{w}_{s}' for w, s in zip(text, pre)])


def f1_score_evaluation(batch_masks, batch_labels, batch_prediction, index_tag):
    all_prediction, all_labels = [], []
    for i in range(len(batch_masks)):
        max_length = batch_masks[i].shape[0]
        for index in range(max_length):
            # 得到句子原始长度
            length = sum(batch_masks[i][index].cpu().numpy() == 1)
            # 截断 原始 label
            _label = batch_labels[i][index].cpu().numpy().tolist()[:length]
            # label -> tag
            label_tag = [index_tag[i] for i in _label]

            # 截断预测 label
            _predict = batch_prediction[i][index][:length]
            # label -> tag
            predict_pred = [index_tag[i] for i in _predict]

            if len(_label) == len(_predict) and index < max_length:
                all_labels.append(label_tag)
                all_prediction.append(predict_pred)

    score = f1_score(all_prediction, all_labels)
    return score


if __name__ == "__main__":
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_data, train_tag, word_2_index, tag_2_index = build_corpus("train", args.train_file, make_vocab=True)
    dev_data, dev_tag = build_corpus("dev", args.dev_file, make_vocab=False)
    test_data, test_tag = build_corpus("test", args.test_file, make_vocab=False)

    train_dataset = MyDataset(train_data, train_tag, word_2_index, tag_2_index)
    train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True,
                              collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, word_2_index, tag_2_index)
    dev_loader = DataLoader(dev_dataset, args.dev_test_batch_size, shuffle=False,
                            collate_fn=dev_dataset.pro_batch_data)

    test_dataset = MyDataset(test_data, test_tag, word_2_index, tag_2_index)
    test_loader = DataLoader(test_dataset, args.dev_test_batch_size, shuffle=False,
                             collate_fn=test_dataset.pro_batch_data)

    pkl.dump([word_2_index, tag_2_index, [i for i in tag_2_index]], open(args.data_pkl, "wb"))

    model = BiLSTM_CRF(len(word_2_index), tag_2_index, args.embedding_num, args.hidden_num, word_2_index["<PAD>"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)
    model = model.to(device)

    score_max = float("-inf")
    for epoch in range(args.epochs):
        save_flag = False
        model.train()
        for batch_data, batch_tag, batch_mask in train_loader:
            batch_data, batch_tag, batch_mask = batch_data.to(device), batch_tag.to(device), batch_mask.to(device)
            loss = model(batch_data, batch_tag, batch_mask)

            model.zero_grad()

            loss.backward()
            opt.step()

        model.eval()
        all_pre, all_tag, all_maks = [], [], []
        with torch.no_grad():
            for batch_data, batch_tag, batch_mask in dev_loader:
                batch_data, batch_tag, batch_mask = batch_data.to(device), batch_tag.to(device), batch_mask.to(device)
                y_pred = model(batch_data, mask=batch_mask)

                all_pre.append(y_pred)
                all_tag.append(batch_tag)
                all_maks.append(batch_mask)

        score = f1_score_evaluation(batch_masks=all_maks, batch_labels=all_tag, batch_prediction=all_pre,
                                    index_tag=[i for i in tag_2_index])

        if score_max < score:
            score_max = score
            torch.save(model.state_dict(), args.save_model_best)
            save_flag = True
        print(
            f"epoch:[{epoch+1}/{args.epochs}]\ttrain_loss:{loss.item():7.3f}\tdev f1_score:{score:.3f}\tsave model:{save_flag}")

    model.eval()
    torch.save(model.state_dict(), args.save_model_last)

    all_pre, all_tag, all_maks = [], [], []
    with torch.no_grad():
        for batch_data, batch_tag, batch_mask in test_loader:
            batch_data, batch_tag, batch_mask = batch_data.to(device), batch_tag.to(device), batch_mask.to(device)
            y_pred = model(batch_data, mask=batch_mask)

            all_pre.append(y_pred)
            all_tag.append(batch_tag)
            all_maks.append(batch_mask)

    score = f1_score_evaluation(batch_masks=all_maks, batch_labels=all_tag, batch_prediction=all_pre,
                                index_tag=[i for i in tag_2_index])

    print(f"test f1 score:{score:.4f}")
