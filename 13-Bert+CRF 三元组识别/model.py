# -*- coding:utf-8 -*-
# @author: 木子川
# @Email:  m21z50c71@163.com
# @VX：fylaicai

import re
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from utils import START_TAG, STOP_TAG, tag2idx, idx2tag, relation2idx, idx2relation

device = "cuda" if torch.cuda.is_available() else 'cpu'


def log_sum_exp(smat):
    """
    for example:
    输入:
    tensor([[[0.5840, 0.6834, 0.8859, 0.6457],
         [0.3828, 0.6881, 0.3363, 0.3396],
         [0.9382, 0.5262, 0.4825, 0.4868]],
        [[0.3437, 0.0670, 0.6303, 0.8735],
         [0.2810, 0.3536, 0.8671, 0.1565],
         [0.4990, 0.4223, 0.2033, 0.6486]]])
    输出:
    tensor([[[1.7604, 1.7339, 1.6947, 1.5972]],
        [[1.4774, 1.3910, 1.7017, 1.7007]]])
    """
    vmax = smat.max(dim=1, keepdim=True).values
    return (smat - vmax).exp().sum(axis=1, keepdim=True).log() + vmax


def entity_average(hidden_output, e_mask):
    # hidden_output: [batch size, seq len, hidden_dim]
    # e_mask: [batch size, seq len]
    e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch size, 1, seq len]
    length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
    sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [batch size, hidden_dim]
    avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    return avg_vector  # [batch size, hidden_dim]


class BertForRE(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForRE, self).__init__(config)
        # 加载 BERT 模型
        self.bert = BertModel(config)
        # 设置 BERT 模型的隐藏维度
        self.hidden_dim = config.hidden_size

        # NER
        # 加载标签索引字典
        self.tag2idx = tag2idx
        # 设置标签的数量
        self.tag_size = len(tag2idx)
        # 定义一个线性层，将隐藏维度映射到标记维度
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        # 定义 NER 的转移矩阵 ，表示从某一列的tag转移至某一行的tag的TransitionScore
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        # 将 START_TAG 的转换得分设置为 -10000
        self.transitions.data[tag2idx[START_TAG], :] = -10000
        # 将 STOP_TAG 的转换得分设置为 -10000
        self.transitions.data[:, tag2idx[STOP_TAG]] = -10000

        # Relation predict
        # 设定关系的数量
        self.target_size = len(relation2idx)
        # 定义一个 Linear 层
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)
        # 定义一个 dropout 层
        self.drop = nn.Dropout(0.1)
        # 定义一个激活函数
        self.activation = nn.Tanh()
        # 定义一个层标准化层
        self.norm = nn.LayerNorm(self.hidden_dim * 3)
        # 定义一个 Linear 层，将隐藏维度映射到标签集维度
        self.hidden2label = nn.Linear(self.hidden_dim * 3, self.target_size)
        # 定义一个损失函数
        self.criterion = nn.CrossEntropyLoss()

    def get_features(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output
        return sequence_output, pooled_output

    def get_total_scores(self, frames, real_lengths):
        """
        frames：[batch size, seq len, tag_size]
        real_lengths：[batch size]
        得到所有可能路径的分数总和
        """
        alpha = torch.full((frames.shape[0], self.tag_size), -10000.0).to(device)  # [batch size, tag_size]
        alpha[:, self.tag2idx[START_TAG]] = 0.  # 初始状态的EmissionScore. START_TAG是0, 其他都是很小的值 "-10000"
        alpha_ = torch.zeros((frames.shape[0], self.tag_size)).to(device)  # [batch size, tag_size]
        frames = frames.transpose(0, 1)  # [seq len, batch size, tag_size]
        index = 0
        for frame in frames:
            index += 1
            # alpha.unsqueeze(-1):当前各状态的分值分布,[batch size, tag_size, 1]
            # frame.unsqueeze(1):发射分值,[batch size, 1, tag_size]
            # self.transitions.T:转移矩阵,[tag_size, tag_size]
            # 三者相加会广播,维度为[batch size, tag_size, tag_size], log_sum_exp后的维度为[batch size, 1, tag_size]
            # alpha [batch size, tag_size]
            alpha = log_sum_exp(alpha.unsqueeze(-1) + frame.unsqueeze(1) + self.transitions.T).squeeze(1)
            for idx, length in enumerate(real_lengths):
                if length == index:
                    alpha_[idx] = alpha[idx]
        # 最后转到EOS，发射分值为0，转移分值为 self.transitions[[self.tag2idx[STOP_TAG]], :].T
        # alpha.unsqueeze(-1): [batch size, tag_size, 1]
        # self.transitions[[self.tag2idx[STOP_TAG]], :].T: [tag_size, 1]
        # 三者相加会广播,维度为[batch size, tag_size, 1], log_sum_exp后的维度为[batch size, 1, 1]
        # [batch_size]
        alpha_ = log_sum_exp(alpha_.unsqueeze(-1) + 0 + self.transitions[[self.tag2idx[STOP_TAG]], :].T).flatten()
        return alpha_

    def get_golden_scores(self, frames, tag_ids_batch, real_lengths):
        """
        frames[batch size, seq len, tag_size]
        tag_ids_batch:[batch size, seq len]
        real_lengths：[batch size]
        得到正确路径的得分
        """
        score = torch.zeros(tag_ids_batch.shape[0]).to(device)  # [batch_size]
        score_ = torch.zeros(tag_ids_batch.shape[0]).to(device)  # [batch_size]
        tags = torch.cat([torch.full([tag_ids_batch.shape[0], 1], self.tag2idx[START_TAG], dtype=torch.long).to(device),
                          tag_ids_batch], dim=1)  # [batch_size, seq_len+1],注意不要+[STOP_TAG]; 结尾有处理
        index = 0
        for i in range(frames.shape[1]):  # 沿途累加每一帧的转移和发射
            index += 1
            frame = frames[:, i, :]  # [batch_size, tag_size]
            score += self.transitions[tags[:, i + 1], tags[:, i]] + frame[
                range(frame.shape[0]), tags[:, i + 1]]  # [batch_size]

            for idx, length in enumerate(real_lengths):
                if length == index:
                    score_[idx] = score[idx]
        score_ = score_ + self.transitions[self.tag2idx[STOP_TAG], tags[:, -1]]  # [batch_size],加上到STOP_TAG的转移
        return score_

    def viterbi_decode(self, frames):
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.tag_size), -10000.).to(device)
        alpha[0][self.tag2idx[START_TAG]] = 0
        for frame in frames:
            # 这里跟get_total_scores稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = alpha.T + frame.unsqueeze(0) + self.transitions.T
            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)
        # 回溯路径
        smat = alpha.T + 0 + self.transitions[[self.tag2idx[STOP_TAG]], :].T
        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1]  # 返回最优路径分值和最优路径

    def get_relation_logit(self, pooled_output, hidden_output, sub_mask, obj_mask):
        """
        关系
        """
        # sub和obj向量的平均值
        sub_h = entity_average(hidden_output, sub_mask)
        obj_h = entity_average(hidden_output, obj_mask)
        sub_h = self.activation(self.dense(sub_h))
        obj_h = self.activation(self.dense(obj_h))
        # [cls] + sub + obj
        concat_h = torch.cat([pooled_output, sub_h, obj_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2label(self.drop(concat_h))
        return logits

    def compute_loss(self, input_ids, attention_mask, tag_ids, sub_mask, obj_mask, labels, real_lengths):
        hidden_output, pooled_output = self.get_features(input_ids, attention_mask)
        feats = self.hidden2tag(hidden_output)
        total_scores = self.get_total_scores(feats, real_lengths)
        gold_score = self.get_golden_scores(feats, tag_ids, real_lengths)
        ner_loss = torch.mean(total_scores - gold_score)
        relation_logits = self.get_relation_logit(pooled_output, hidden_output, sub_mask, obj_mask)
        relation_loss = self.criterion(relation_logits, labels)
        return ner_loss + relation_loss

    @staticmethod
    def extract(chars, tags):
        result = []
        pre = ''
        w = []
        subjects = set()
        objects = set()
        for idx, tag in enumerate(tags):
            if not pre:
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
            else:
                if tag == f'I-{pre}':
                    w.append(chars[idx])
                else:
                    result.append([w, pre])
                    w = []
                    pre = ''
                    if tag.startswith('B'):
                        pre = tag.split('-')[1]
                        w.append(chars[idx])
        for res in result:
            if res[1] in {'SUB', 'BOTH'}:
                subjects.add(''.join(res[0]))
            if res[1] in {'OBJ', 'BOTH'}:
                objects.add(''.join(res[0]))
        return subjects, objects

    def predict(self, text, chars, input_ids, attention_mask):
        """
        text:经过 lower预处理
        chars：[seq_len]；ex:['[CLS]','例','子',...,'[SEP]']
        input_ids: [1, seq_len]
        attention_mask: [1, seq_len]
        """
        hidden_output, pooled_output = self.get_features(input_ids, attention_mask)
        feats = self.hidden2tag(hidden_output)
        feats = feats.squeeze(0)  # [seq len, tag_size]
        res_ner = self.viterbi_decode(feats)
        pred_tags = [idx2tag[ix] for ix in res_ner[1]]
        # 得到两个实体
        subjects, objects = self.extract(chars, pred_tags)
        print(f'Subject包含：{subjects}')
        print(f'Object包含：{objects}')

        # 判断关系
        length = len(text)
        for sub in subjects:
            for obj in objects:
                if obj != sub:
                    try:
                        # 得到sub_mask:[1, seq len]
                        sub_mask = [0] * length
                        start_sub, end_sub = re.search(sub, text).span()
                        sub_mask[start_sub:end_sub] = [1] * (end_sub - start_sub)
                        sub_mask = [0] + sub_mask + [0]
                        sub_mask = torch.tensor(sub_mask, dtype=torch.long).unsqueeze(0).to(device)

                        # 得到obj_mask:[1, seq len]
                        obj_mask = [0] * length
                        start_obj, end_obj = re.search(obj, text).span()
                        obj_mask[start_obj:end_obj] = [1] * (end_obj - start_obj)
                        obj_mask = [0] + obj_mask + [0]
                        obj_mask = torch.tensor(obj_mask, dtype=torch.long).unsqueeze(0).to(device)

                        # 关系 [1, target_size]
                        logits = self.get_relation_logit(pooled_output, hidden_output, sub_mask, obj_mask)

                        relation = idx2relation[logits.argmax().item()]
                        print(f'{sub} - {relation} - {obj}')
                    except:
                        print(f'{sub} - {obj}: 向量提取异常')
