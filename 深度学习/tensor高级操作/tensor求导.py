# https://blog.csdn.net/douhaoexia/article/details/78821428
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parameter as Parameter

w1 = torch.Tensor([2]) #认为w1 与 w2 是函数f1 与 f2的参数
w1 = Variable(w1,requires_grad=True)
w2 = torch.Tensor([2])
w2 = Variable(w2,requires_grad=True)
x2 = torch.rand(1)
x2 = Variable(x2,requires_grad=True)
y2 = x2**w1            # f1 运算
z2 = w2*y2+1           # f2 运算
z2.backward()
print(x2.grad)
print(y2.grad)
print(w1.grad)
print(w2.grad)