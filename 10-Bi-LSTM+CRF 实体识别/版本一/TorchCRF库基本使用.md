# 安装
```
pip install pytorch-crf -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 初始化
```
# 初始化
from torchcrf import CRF

def __init__(self, num_tags: int, batch_first: bool = False) -> None:
    super().__init__()
    self.num_tags = num_tags
    self.batch_first = batch_first

num_tags: 类别数 Number of tags. 
batch_first: 批次数是否在第一维度 Whether the first dimension corresponds to the size of a minibatch. 
```

## loss
```     
def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = 'sum',
) -> torch.Tensor:

功能：计算给定发射分数的标记序列的条件对数似然
参数含义：
emissions：发射矩阵（各标签的预测值）(seq_length, batch_size, num_tags) 
if batch_first is False else (batch_size, seq_length, num_tags)
tags：小批量中每个序列的答案标签（batch_size，seq_len）
mask：掩码张量，形状为(seq_length, batch_size)，如果"batch_first=True"，则形状为(batch_size, seq_length) 
reduction：指定在输出时应用的归约方式。
如果设置为"none"，则不会应用任何归约。
如果设置为"sum"，则输出将在批次上求和。
如果设置为"mean"，则输出将在批次上求平均值。
如果设置为"token_mean"，则输出将在令牌上求平均值
return: 返回值是对数似然值(batch_size,)。如果设置的归约方式为"none"，则返回的张量形状为(batch_size,)；否则形状为()。      
```

## 解码
```
def decode(self, emissions: torch.Tensor,
           mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
         
功能：使用维特比算法找到最可能的标记序列  
参数含义：
emissions：发射矩阵（各标签的预测值）(seq_length, batch_size, num_tags) 
if batch_first is False else (batch_size, seq_length, num_tags)
mask：掩码张量，形状为(seq_length, batch_size)，如果"batch_first=True"，则形状为(batch_size, seq_length)
return: 每个批次中最佳标记序列的列表

```
