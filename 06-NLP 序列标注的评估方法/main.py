# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score
from seqeval.metrics import classification_report

y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER']]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"accuracy：{accuracy:0.4f}\tprecision：{precision:.4f}\trecall：{recall:.4f}\tf1：{f1:.4f}")
print(f"评估报告：{report}")
