模型都未进行调参，未能使模型的准确率达到最高
# 项目名称：
使用 LSTM 模型来对进行实体识别

# 项目环境：
pytorch、python   
相关库安装
```
pip install -r requirement.txt
```

# 项目目录：
```
LSTM
    |--data             数据
    |--model            保存的模型
    |--img              存放模型相关图片 
    |--config.py        配置文件
    |--main.py          主函数
    |--model.py         模型文件
    |--predict.py       预测文件
    |--requirement.txt  安装库文件
    |--utils.py         数据处理文件
```

# 模型介绍
LSTM(Long_short_term_memory),使用LSTM模型可以更好的捕捉到较长距离的依赖关系，通过训练可以学到记忆那些信息和遗忘那些信息，
能解决梯度爆炸和梯度弥散问题，可以处理更长的文本数据。         
遗忘门（记忆门）：         
输入：前一时刻的隐层状态![./img/h_(t-1).png](./img/h_(t-1).png)，当前时刻的输入词![./img/x_t.png](./img/x_t.png)    
输出：遗忘门的值![./img/f_t.png](./img/f_t.png)        
![./img/遗忘门公式.png](./img/遗忘门公式.png)       
输入门：     
输入：前一时刻的隐层状态![./img/h_(t-1).png](./img/h_(t-1).png)，当前时刻的输入词![./img/x_t.png](./img/x_t.png)    
输出：记忆门的值![./img/x_t.png](./img/(C_t%20)%20̃.png)，临时细胞状态![./img/x_t.png](./img/(C_t%20)%20̃.png)
![./img/输入门公式1.png](./img/输入门公式1.png)     
![./img/输入门公式2.png](./img/输入门公式2.png)     
输出门：     
输入：前一时刻的隐层状态![./img/h_(t-1).png](./img/h_(t-1).png)，当前时刻的输入词![./img/x_t.png](./img/x_t.png)，当前时刻细胞状态![./img/C_t.png](./img/C_t.png)        
输出：输出们的![./img/O_t.png](./img/O_t.png)，隐层状态![./img/h_t.png](./img/h_t.png)
![./img/输出门公式1.png](./img/输出门公式1.png)     
![./img/输出门公式2.png](./img/输出门公式2.png)

[LSTM 模型视频讲解](https://www.bilibili.com/video/BV1if4y1x7vf?p=8&vd_source=946c3fb8fdb85266f0efe9377f81df78)

# 项目数据集
数据集用的是论文[【ACL 2018Chinese NER using Lattice LSTM】](https://github.com/jiesutd/LatticeLSTM)中从新浪财经收集的简历数据。

# 模型训练
`python main.py`

# 模型预测
`python predict.py`

# 博客地址

[CSDN LSTM 文本实体识别](https://blog.csdn.net/qq_48764574/article/details/126070478)

[知乎 LSTM 文本实体识别](https://zhuanlan.zhihu.com/p/644160458)

[微信公众号 LSTM 文本实体识别](https://mp.weixin.qq.com/s?__biz=MzkxOTUzMDE0Nw==&mid=2247485314&idx=2&sn=7cb65d250332dcb0c8112bfb0e5167a2&chksm=c1a1f84ef6d67158c5ea20d659338a7162f3f2b92410ef6542fce7b69176bcd0c4645fb73fcb&scene=178&cur_album_id=3109690569678979074#rd)

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
