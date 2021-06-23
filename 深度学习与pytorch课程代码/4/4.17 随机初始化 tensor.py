import torch
from torch import tensor
import numpy as np
# 1 rand()使用[0,1]随机分布
a1 = torch.rand(3,3)
print(f"1 rand {a1} \n")

# 2 like使用别的tensor的shape创建一个tensor
a2 = torch.rand_like(a1)
print(f"2 like使用别的tensor的shape创建一个tensor a2 = {a2} \n")

# 3 randint  最小值，最大值，shape
a3 = torch.randint(10,100,[4,3])
print(f"3 randint  最小值，最大值，shape a3 = {a3} \n")

# 4 正态分布  均值为0，方差为1，即高斯白噪声
a4 = torch.randn(3,3)
print(f"4 正态分布 均值为0，方差为1，即高斯白噪声 a4 = {a4} \n")

# 5 full
a5 = torch.full([3,4],7)
print(f"5 full a5 = {a5} \n")

# 6 随机梯度下降 arange
a6 = torch.arange(0,10,dtype = float)
print(f"6 arange a6 = {a6} \n")

# 7 神经网络与全连接层 linspace[0,10]
a7 = torch.linspace(0,10,steps = 10)
print(f"7 linspace a7 = {a7} \n")

# 8 logspace[0,10]    0至-4 切成10份，然后以log2为底求
a8 = torch.logspace(0,-1,steps = 10,base = 2)
print(f"8 logspace[0,10] a8= {a8} \n")

# 9 卷积神经网络CNN 生成全部是0，或1,或对角线是1
a9 = torch.ones(3,3)
a10 = torch.zeros(3,3)
a11= torch.eye(3,4)
a12 = torch.ones_like(a11)
print(a9,a10,a11,a12)

# 10随机打散
a13 = torch.randperm(12)
print(a13)