# 项目名称
中文命名实体识别的标注方法，本文只介绍常见的标注方法。

# 项目目录
```
命名实体标注   
    |-- data
        |-- error.txt       错误标记（word_dict.txt中内容不全）
        |-- Tagging.txt     标记文本
        |-- noTagging.txt   未标记文本
        |-- word_dict.txt   需要标注的内容 
    |-- main.py             主函数
```

# 三种标注方法
```
BIO：B-begin,I-inside,O-outside。
B-X代表实体X的开头、I-X代表实体的结尾、O代表不属于任何类型的
BMOES：B-begin,M-middle,E-end,S-single。
B-X表示一个词的词首位值，M-X示一个词的中间位置，E-X表示一个词的末尾位置，S-X表示一个单独的字词
BIOES：B-begin,I-inside,O-outside,E-end,S-single。
B-x表示开始，I-x表示内部，O表示非实体，E-x实体尾部，S-x表示该词本身就是一个实体
```

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
