模型都未进行调参，未能使模型的准确率达到最高
# 项目名称
使用 Word2Vec 来训练字&词向量并使用

# 项目环境
python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录
```
Word2Vec      
    |-- data                     数据集    
    |-- img                      相关图片                          
    |-- main.py                  主函数
    |-- requiremeny.txt          相关库
    |-- word.model               字训练模型
    |-- word_data.vector         字向量
    |-- WordPartialWeight.pkl    单独保存的字的数据
    |-- words.model              词训练模型
    |-- words_data.vector        词向量
    |-- WordsPartialWeight.pkl   单独保存的词的数据
```

# 项目数据集
数据集使用THUCNews中的test.txt中所有的文本数据

# 模型训练与查看
`python main.py`

# 备注
本模型在训练时并没有调参到最优，因此会有误差

# 博客地址
[CSDN Word2Vec 训练字&词向量](https://blog.csdn.net/qq_48764574/article/details/126350812)

[知乎 Word2Vec 训练字&词向量](https://zhuanlan.zhihu.com/p/642943733)

[微信公众号 Word2Vec 训练字&词向量](https://mp.weixin.qq.com/s?__biz=MzkxOTUzMDE0Nw==&mid=2247485314&idx=1&sn=c13a4d87fb125e0fd785e06890cb9d5a&chksm=c1a1f84ef6d67158ff0e8592e773a92089ccec6d508dce92e082c57c2d5c07d7870a9a4fec9b&scene=178&cur_album_id=3109690569678979074#rd)

# 深度学习交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   

