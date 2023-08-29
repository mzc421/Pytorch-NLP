模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：
使用 Bi-LSTM 模型来对进行实体识别

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
BiLSTM
    |--data             数据
    |--img              存放模型相关图片 
    |--model            保存的模型
    |--config.py        配置文件
    |--main.py          主函数
    |--model.py         模型文件
    |--predict.py       预测文件
    |--requirement.txt  安装库文件
    |--utils.py         数据处理文件
```

# 模型介绍
Bi-LSTM(Bi-directional LSTM)，就可以更好的捕捉双向的语义依赖，对于更细粒的分类可以很好学到（如：表示程度的词）。      
由前向的 LSTM 和 后向的 LSTM 结合成 Bi-LSTM      
决定参数：```bidirectional: If True, becomes a bidirectional LSTM. Default: False```

# 项目数据集
数据集用的是论文[【ACL 2018Chinese NER using Lattice LSTM】](https://github.com/jiesutd/LatticeLSTM)中从新浪财经收集的简历数据。

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
