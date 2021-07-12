import numpy as np
import torch

a1 = torch.arange(8).reshape(2,4).float()
print(a1)
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6 随机梯度下降., 7 神经网络与全连接层.]])
a = torch.ones_like(a1)
b = torch.zeros_like(a1)
a1 = torch.where(a1>5,a,b)
print(a1)