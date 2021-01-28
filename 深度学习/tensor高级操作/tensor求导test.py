# https://blog.csdn.net/douhaoexia/article/details/78821428
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parameter as Parameter

w1 = torch.tensor([3.],requires_grad=True) #认为w1 与 w2 是函数f1 与 f2的参数

w2 = torch.tensor([2.],requires_grad=True)


# x2 = torch.tesnor([1.],requires_grad=True)
            # f1 运算
z2 = w2**w1+1           # f2 运算
z2.backward()
print(z2)
