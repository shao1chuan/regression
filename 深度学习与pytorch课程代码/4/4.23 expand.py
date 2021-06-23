import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(1,32,1,1)
print(a1.expand(4,32,2,2).shape)

a1 = torch.rand(4,32,8,8)
a2 = torch.rand(1,32,1,1)
a3 = a2.expand_as(a1)