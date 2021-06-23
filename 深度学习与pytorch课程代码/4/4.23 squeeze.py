import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(1,32,1,1)
print(a1.squeeze(-1).shape,a1.squeeze().shape)