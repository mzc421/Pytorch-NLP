模型都未进行调参，未能使模型的准确率达到最高

# 项目名称：
使用 Bert+CRF 模型来对进行实体识别，
借鉴Bert模型的输出和掩码进行关系识别

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
BiLSTM
    |--bert-base-chinese             bert模型权重文件
    |--data                          数据
    |--img                           存放模型相关图片 
    |--model                         保存的模型
    |--config.py                     配置文件
    |--main.py                       主函数
    |--model.py                      模型文件
    |--predict.py                    预测文件
    |--requirement.txt               安装库文件
    |--split_data.py                 数据划分
    |--utils.py                      数据处理文件
```

# 模型介绍
在实体识别中：使用了Bert模型，CRF模型
在关系识别中：使用了Bert模型的输出与实体掩码，进行一系列变化，得到关系

# 项目数据集
[数据集](https://github.com/buppt//raw/master/data/people-relation/train.txt)
朱时茂	陈佩斯	合作	《水与火的缠绵》《低头不见抬头见》《天剑群侠》小品陈佩斯与朱时茂1984年《吃面条》合作者：陈佩斯聽1985年《拍电影》合
女	卢润森	unknown	卢恬儿是现任香港南华体育会主席卢润森的千金，身为南华会太子女的卢恬儿是名门之后，身家丰厚，她长相
侯佩岑	黄伯俊	夫妻	场照片事后将发给媒体，避免采访时出现混乱，[3]举行婚礼侯佩岑黄伯俊婚纱照2011年4月17日下午2点，70名亲友见证下，侯佩
李敖	王尚勤	夫妻	李敖后来也认为，“任何当过王尚勤女朋友的人，未来的婚姻都是不幸的！
傅家俊	丁俊晖	好友	改写23年历史2010年10月29日，傅家俊1-5输给丁俊晖，这是联盟杯历史上首次出现了中国德比，丁俊晖傅家俊携手改写了


# 数据划分
`python split_data.py`

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 博客地址

[CSDN Bert_CRF 三元组识别](https://blog.csdn.net/qq_48764574/article/details/132344244)

[知乎 Bert_CRF 三元组识别](https://zhuanlan.zhihu.com/p/651639890)

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
