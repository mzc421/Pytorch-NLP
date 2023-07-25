模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：
使用 Bert 模型来对进行实体识别

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
Bert 实体识别
    |-- bert-base-chinese    bert 中文预训练模型     
    |-- data                 数据集  
    |-- img                  存放模型相关图片              
    |-- model                保存的模型               
    |-- config.py            配置文件                                 
    |-- main.py              主函数                      
    |-- model.py           模型文件                     
    |-- predict.py           预测文件                         
    |-- requirement.txt      需要的安装包
    |-- utils.py             数据处理文件
```

# 项目数据集
数据集用的是论文[【ACL 2018Chinese NER using Lattice LSTM】](https://github.com/jiesutd/LatticeLSTM)中从新浪财经收集的简历数据。

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# Bert 文本分类 与 Bert 实体识别 的区别
文本分类部分代码
```
hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                       output_hidden_states=False)  # 控制是否输出所有encoder层的结果
# shape (batch_size, hidden_size)  pooler_output -->  hidden_out[0]
pred = self.linear(hidden_out.pooler_output)
```
实体识别部分代码
```
bert_out = self.bert(batch_index)
bert_out0, bert_out1 = bert_out[0], bert_out[1]
pre = self.classifier(bert_out0)
```
可以看到在送入线性层的数据不同，一个是`last_hidden_state(batch_size, sequence_length, hidden_size)`，
另一个是`pooler_output(batch_size, hidden_size)`。          
last_hidden_state：模型最后一层输出的隐藏状态序列。     
pooler_output：在通过用于辅助预训练任务的层进行进一步处理后，序列的第一个标记（分类标记）的最后一层隐藏状态。
对于 BERT 系列模型，这会在通过线性层和 tanh 激活函数处理后返回分类标记。线性层权重在预训练期间从下一个句子预测（分类）目标进行训练。         
相当于 `last_hidden_state` 用于字符级别，一句文本里面有`sequence_length`个字，每个字用`hidden_size`的一部分表示；
`pooler_output` 用于篇章级别（分类），一句文本用`hidden_size`表示

# QQ交流群
![QQ群](img/QQ群.jpg)   