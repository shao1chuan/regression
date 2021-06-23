import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(4,3,28,28)
print(a1[0].shape,a1[0][0].shape,a1[0][0][0][0])
# 4 表示前2张图片，第1通道的所有数据
print(f"1 表示前2张图片，第1通道的所有数据 a1[:2,:1,].shape = {a1[:2,:1,].shape}")
# 2 各行采样
print(f"{a1[:,:,0:28:2,0:28:2].shape}")
print(f"{a1[:,:,::2,::2].shape}")