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
    |--saved_model                         保存的模型
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

# 数据划分
`python split_data.py`

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# QQ交流群
![QQ群](img/QQ群.jpg)   
