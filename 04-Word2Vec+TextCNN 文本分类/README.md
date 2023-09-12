模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：
使用 Word2Vec-TextCNN 模型来对中文进行分类，即文本分类

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
Word2Vec-TextCNN         
    |-- data                 数据集   
    |-- img                  存放相关图片
    |-- model                保存的模型               
    |-- config.py            配置文件                    
    |-- main.py              主函数                      
    |-- model.py             模型文件                     
    |-- predict.py           预测文件                         
    |-- requirement.txt      需要的安装包   
    |-- TextCNN.pdf          TextCNN 的论文
    |-- utils.py             数据处理文件
   ```

# 项目数据集
数据集使用THUCNews中的train.txt、test.txt、dev.txt，为十分类问题。
其中训练集一共有 180000 条，验证集一共有 10000 条，测试集一共有 10000 条。
其类别为 finance、realty、stocks、education、science、society、politics、sports、game、entertainment 这十个类别。

# 模型介绍
详细内容请看：[TextCNN 文本分类介绍](../01-TextCNN%20文本分类/README.md)

# 修改部分
相对于原始 TextCNN 模型的 Emdedding 层，此项目用了 Word2Vec 来代替。
关于 Word2Vec 训练得到词向量，可以看：[Word2Vec 字&词向量](../00-Word2Vec%20字&词向量)

```
# 添加 "<pad>" 和 "<UNK>"
# {"<PAD>": np.zeros(self.embedding), "<UNK>": np.random.randn(self.embedding)}
self.Embedding = self.model.vectors
self.Embedding = np.insert(self.Embedding, self.num_word, [np.zeros(self.embedding), np.random.randn(self.embedding)], axis=0)

self.word_2_index = self.model.key_to_index
self.word_2_index.update({"<PAD>": self.num_word, "<UNK>": self.num_word + 1})
```

```
text_id = [self.word_2_index.get(i, self.word_2_index["<UNK>"]) for i in text]
        text_id = text_id + [self.word_2_index["<PAD>"]] * (self.max_len - len(text_id))

wordEmbedding = np.array([self.Embedding[i] for i in text_id])

text_id = torch.tensor(wordEmbedding).unsqueeze(dim=0)
```

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   


