import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(4,3,32,32)
a2 = a1.transpose(1,3)

# 4,32,32,3
a3  =a2.contiguous().reshape(4,32*32*3).reshape(4,32,32,3).transpose(1,3)
print(a3.shape)

# 验证是否一致
print(torch.all(a1.eq(a3)))

a4 = a1.permute(0,2,3,1)
a5 = a1.transpose(1,2).transpose(2,3)
print(torch.all(a4.eq(a5)))