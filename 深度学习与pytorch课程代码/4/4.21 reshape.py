import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(4,3,28,28)

print(f" = {a1.reshape(4,3*28,28).shape}")


