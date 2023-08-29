# 项目名称：
命名实体等任务的评估方法。         
词性标注、命名实体识别等 NLP 任务都是属于序列标注类型的任务的，本质属于分类任务。
对于序列标注类型的模型的结果评估也有对应的模块实现，模块名叫 seqeval。            
[GitHub seqeval](https://github.com/chakki-works/seqeval)

# 项目环境：
python             
安装库：
```
pip install -r requirement.txt
```   

# 支持的标注方式：
seqeval 支持 BIO，IOBES 标注模式，可用于命名实体识别，词性标注，语义角色标注等任务的评估。

# 使用方法：
同 sklearn 库     

# 计算方法：
```
准确率: accuracy = 预测对的元素个数/总的元素个数           
查准率：precision = 预测正确的实体个数 / 预测的实体总数      
召回率：recall = 预测正确的实体个数 / 预测中标注的实体总数
F1值：F1 = 2 *准确率 * 召回率 / (准确率 + 召回率)
```

```
y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER']]

accuracy：0.6667	precision：0.3333	recall：0.5000	f1：0.4000
评估报告：              precision    recall  f1-score   support

        MISC       0.00      0.00      0.00         1
         PER       1.00      1.00      1.00         1

   micro avg       0.33      0.50      0.40         2
   macro avg       0.50      0.50      0.50         2
weighted avg       0.50      0.50      0.50         2
```

```
accuracy：
准确率: accuracy = 预测对的元素个数/总的元素个数 
    总数量数量：9
    预测正确的数量：6
    accuracy = 6 / 9 = 0.6667
所以在序列标注任务中，以准确率来进行判断模型的好坏是不准确的。
```

```
precision：
查准率：precision = 预测正确的实体个数 / 预测的实体总数 
    预测正确的实体个数：3
    预测的实体总个数：9
    accuracy = 3 / 9 = 0.3333
```

```
recall：
召回率：recall = 预测正确的实体个数 / 预测中标注的实体总数
    预测正确的实体个数：3
    标注的实体总个数：6
    recall = 3 / 6 = 0.5
```

```
f1：越大越好  最佳为1，最差为0
F1值：F1 = 2 * 准确率 * 召回率 / (准确率 + 召回率)
F1 = (2 * 0.3333 * 0.5) / (0.3333 + 0.5) = 0.4
```

# 微信交流群
我们有一个微信交流群，大家如果有需要，可以加入我们，一起进行学习。关注公众号后会有一个私人微信，添加微信，备注进群，就可以拉你进群，进行学习。

![公众号](img/公众号.jpg)   
