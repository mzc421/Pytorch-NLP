模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：

使用 Bert + TextCNN 融合模型来对中文进行分类，即文本分类
Bert往往可以对一些表述隐晦的句子进行更好的分类，TextCNN往往对关键词更加敏感。

# 项目环境：

pytorch、python   
相关库安装

```
pip install -r requirement.txt
```

# 项目目录：

```
bert-TextCNN  
    |-- bert-base-chinese    bert 中文预训练模型     
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

# bert-TextCNN 模型结构图

## 模型1

![bert-TextCNN 模型图2](img/bertTextCnn模型图2.png)
将 bert 模型的最后一层的输出的内容作为 TextCNN 模型的输入，送入模型在继续进行学习，得到最终的结果，进行文本分类

## 模型2

![bert-TextCNN 模型图1](img/bertTextCnn模型图1.png)        
Bert-Base除去第一层输入层，有12个encoder层，每个encode层的第一个token（CLS）向量都可以当作句子向量，
我们可以抽象的理解为，encode层越浅，句子向量越能代表低级别语义信息，越深，代表更高级别语义信息。
我们的目的是既想得到有关词的特征，又想得到语义特征，模型具体做法是将第1层到第12层的CLS向量，作为TextCNN的输入，进行文本分类。

# 项目数据集

数据集使用THUCNews中的train.txt、test.txt、dev.txt，为十分类问题。
其中训练集一共有 180000 条，验证集一共有 10000 条，测试集一共有 10000 条。
其类别为 finance、realty、stocks、education、science、society、politics、sports、game、entertainment 这十个类别。

# 模型训练

`python main.py`

# 模型预测

`python predict.py`

# 训练自己的数据集

train.txt、dev.txt、test.txt 的数据格式：文本\t标签（数字表示）

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

[CSDN Bert-TextCNN 文本分类](https://blog.csdn.net/qq_48764574/article/details/126323731)

[知乎 Bert-TextCNN 文本分类](https://zhuanlan.zhihu.com/p/642209326)

[微信公众号 Bert-TextCNN 文本分类](https://mp.weixin.qq.com/s?__biz=MzkxOTUzMDE0Nw==&mid=2247485111&idx=1&sn=ca93eaef75b3351b3084570a453d9006&chksm=c1a1f97bf6d6706dd800819ebab041ce45777dac568b833908efbb2c8856bd71abecb990f0a9&scene=178&cur_album_id=3109690569678979074#rd)

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
