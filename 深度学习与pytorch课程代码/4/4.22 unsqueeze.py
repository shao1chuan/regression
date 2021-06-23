import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(4,3,28,28)

# 在a1前面插入一个维度,增加一个图片分组  正数在前面
a2 = a1.unsqueeze(0)
print(f"在a1前面插入一个维度,增加一个图片分组 = {a2.shape}")
# 在a1后面面插入一个维度,增加一个图片分组  负数在后面
a3 = a1.unsqueeze(-1)
print(f"在a1后面面插入一个维度,增加一个图片分组 = {a3.shape}")


a4 = torch.tensor([1,2])
a5 = a4.unsqueeze(-1)
a6 = a5.unsqueeze(0)
print(f"a4.a5.a6.shape = {a4.shape,a5.shape,a6.shape}",a4,a5,a6)

# 真实案例
a7 = torch.rand(32)
# 需要变成(4,32,14,14)
a8 = a7.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
print(a8.shape)