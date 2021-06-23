import torch
from torch import tensor

# 6 随机梯度下降 掩码
a1 = torch.rand(3,4)
print(a1)
# 把大于0.5的选出来
print(a1.ge(0.5))
print(a1.masked_select(a1.ge(0.5)))

# 7 神经网络与全连接层 打平选取
a2 = torch.tensor([[1,2,3],[4,5,6]])
a3 = a2.take(torch.tensor([1,3,5]))
print(a3)