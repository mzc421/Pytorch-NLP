模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：
使用 bert 模型来对中文进行分类，即文本分类

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
Bert        
    |-- bert-base-chinese    bert 中文预训练模型            
    |-- data                 数据集   
    |-- img                  存放模型相关图片 
    |-- model                保存的模型               
    |-- config.py            配置文件                    
    |-- main.py              主函数                      
    |-- model.py             模型文件                     
    |-- predict.py           预测文件                         
    |-- requests.txt         需要的安装包                
    |-- test.py              测试文件              
    |-- utils.py             数据处理文件
```

# Bert 模型结构与文本分类模型结构
![bert 模型结构](img/bert%20模型结构.jpg)   
![bert 文本分类模型](img/bert%20文本分类模型.jpg)    
`Overall pre-training and fine-tuning procedures for BERT. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize
models for different down-stream tasks. During fine-tuning, all parameters are fine-tuned. [CLS] is a special
symbol added in front of every input example, and [SEP] is a special separator token (e.g. separating questions/answers).`   

`BERT的总体预训练和微调程序。除了输出层之外，在预训练和微调中使用相同的架构。相同的预训练模型参数用于初始化
不同下游任务的模型。在微调期间，将微调所有参数。[CLS]是一个特殊的
符号添加在每个输入示例的前面，[SEP]是一个特殊的分隔符标记（例如，分隔问题/答案）。`

# Bert 模型的预训练和微调结构
![bert Pre-training and Fine-Tuning](img/bert%20Pre-training%20and%20Fine-Tuning.jpg)  
左侧的图表示了预训练的过程，右边的图是对于具体任务的微调过程。

# Bert 模型的输入
BERT 的输入可以包含一个句子对 (句子 A 和句子 B)，也可以是单个句子。同时 BERT 增加了一些有特殊作用的标志位：   
[CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。   
[SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。   
[MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么。   

# Bert 模型的 Embedding 模块
BERT 得到要输入的句子后，要将句子的单词转成 Embedding，Embedding 用 E 表示。  
与 transformer 不同，BERT 的输入 Embedding 由三个部分相加得到：Token Embedding，Segment Embedding，position Embedding。   
![bert Embedding 模块](img/bert%20Embedding模块.jpg)   
Token Embedding：单词的 Embedding，例如 [CLS] dog 等，通过训练学习得到。   
Segment Embedding：用于区分每一个单词属于句子 A 还是句子 B，如果只输入一个句子就只使用 EA，通过训练学习得到。   
position Embedding：编码单词出现的位置，与 transformer 使用固定的公式计算不同，BERT 的 position Embedding 也是通过学习得到的，在 BERT 中，假设句子最长为 512。

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

体验2D巅峰 倚天屠龙记十大创新概览\t8   
60年铁树开花形状似玉米芯(组图)\t5    

class.txt：标签类别（文本）
## 修改内容：
在配置文件中修改长度、类别数、预训练模型地址    
parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese")   
parser.add_argument("--class_num", type=int, default=10)   
parser.add_argument("--max_len", type=int, default=38)

# 博客地址
[CSDN Bert 文本分类](https://blog.csdn.net/qq_48764574/article/details/126068667)

[知乎 Bert 文本分类](https://zhuanlan.zhihu.com/p/641995484)

[微信公众号 Bert 文本分类](https://mp.weixin.qq.com/s?__biz=MzkxOTUzMDE0Nw==&mid=2247485015&idx=1&sn=7c9d28ad97cd075edda39c873f55c43a&chksm=c1a1f99bf6d6708de21ec6a636623378e4bed344a3de075033b555215b2d256f95443c79e00e&token=339959978&lang=zh_CN#rd)

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
