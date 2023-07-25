# 项目名称：
使用 Bi-LSTM+CRF 模型来对进行实体识别

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
Bi-LSTM-CRF  
    |-- data                 数据集   
    |-- img                  存放模型相关图片            
    |-- model                保存的模型               
    |-- config.py            配置文件                              
    |-- main.py              主函数                      
    |-- model.py             模型文件                     
    |-- predict.py           预测文件                         
    |-- requirement.txt      需要的安装包
    |-- utils.py             数据处理文件
```

# 项目介绍：
本项目中使用了三个版本来学习使用 Bi-LSTM+CRF，
版本一 是 pytorch 库中的 torchcrf 来学习使用，具体内容在此：[TorchCRF库基本使用](./版本一/TorchCRF库基本使用.md)         
版本二 是直接从数学逻辑中直接编写 CRF 模块

# 项目数据集
数据集用的是论文[【ACL 2018Chinese NER using Lattice LSTM】](https://github.com/jiesutd/LatticeLSTM)中从新浪财经收集的简历数据。

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 训练自己的数据集
数据格式：文本\t标签（数字表示）

```
体验2D巅峰 倚天屠龙记十大创新概览\t8   
60年铁树开花形状似玉米芯(组图)\t5    
```

class.txt：标签类别（文本）

## 修改内容：
在配置文件中修改长度、类别数、预训练模型地址    

```
parser.add_argument("--select_model_last", type=bool, default=True, help="选择模型")
parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese", help="bert 预训练模型")
parser.add_argument("--class_num", type=int, default=10)   
parser.add_argument("--max_len", type=int, default=38)
```

# 博客地址
[CSDN Bert+TextCNN 文本分类](https://blog.csdn.net/qq_48764574/article/details/126323731)

[知乎 Bert+TextCNN 文本分类](https://zhuanlan.zhihu.com/p/642209326)

# QQ交流群
![QQ群](img/QQ群.jpg)   

