import torch
import torch.nn as nn
# 1 nn

torch.manual_seed(1)
# 输入为 [b,3,28,28]  1:chanel
x = torch.rand(  7,      3,  28,28)
             # 图片数， 通道数
layer = nn.Conv2d(  3,    16,kernel_size=5,stride=1,padding=1)
                # 通道数，核数
out1 = layer.forward(x)
# out = layer(x)  与 out = layer.forward(x)  效果一样
print(f"out1 = {out1.shape}")
# out1 = torch.Size([7 神经网络与全连接层, 16, 26, 26])   x@layer
# print(layer.weight)

# 2 F
import torch.nn.functional as F
x = x.clone()
w = torch.rand(16,3,5,5)
b = torch.rand(16)
out2 = F.conv2d(x,w,b,stride=1,padding=1)
print(f"out2 = {out2.shape}")
# out2 = torch.Size([7 神经网络与全连接层, 16, 26, 26])  x@w.t()

# 1 nn
layer = nn.MaxPool2d(2,stride=2)
out = layer(x)
print(out.shape)
# torch.Size([7 神经网络与全连接层, 3, 14, 14])
# 2 F
out = F.avg_pool2d(out2,2,stride = 2)
print(out.shape)
# torch.Size([7 神经网络与全连接层, 16, 13, 13])