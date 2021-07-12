import numpy as np
import torch

a1 = torch.arange(8).reshape(2,4).float()
print(a1)
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6 随机梯度下降., 7 神经网络与全连接层.]])
print("a1.min(),a1.max(),a1.prod()连乘,a1.mean()",a1.min(),a1.max(),a1.prod(),a1.mean())
# tensor(0.) tensor(7 神经网络与全连接层.) tensor(0.) tensor(3.5000)
print("a1.argmax(),a1.argmin()",a1.argmax(),a1.argmin())
# tensor(7 神经网络与全连接层) tensor(0)

torch.manual_seed(1)
a2 = torch.randn(4,10)


print("a2.argmax(dim=0) ",a2.argmax(dim=0))
#dim=0是在[4,10]第0维上 返回最大所对应的序号
# a2.argmax(dim=0)  tensor([2, 3, 3, 1, 3, 3, 2, 3, 1, 0])


print("a2.argmax(dim=1) ",a2.argmax(dim=1))
# 4张照片，每张概率最大值的索引号【4，10】 dim=1是在[4,10]第二维上，返回最大所对应的序号
# a2.argmax(dim=1)  tensor([9 卷积神经网络CNN, 3, 6 随机梯度下降, 2])

print("a2.argmax(dim=1,keepdim = True) ",a2.argmax(dim=1,keepdim = True))

print("a2.max(dim=1,keepdim = True) ",a2.max(dim=1,keepdim = True))

print("a2.kthvalue(8,dim=1) ",a2.kthvalue(10,dim=1))
# values=tensor([0.3037, 1.6871, 2.0242, 2.4070]),
# indices=tensor([9 卷积神经网络CNN, 3, 6 随机梯度下降, 2]))
# 第10小的 就是第一大的