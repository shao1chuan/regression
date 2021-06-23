import  torch
import  torch.nn as nn
import  torch.nn.functional as F

x = torch.rand(1,16,14,14)
layer = nn.MaxPool2d(kernel_size=2,stride=2)
out = layer(x)
print(out.shape)
# torch.Size([1, 16, 7 神经网络与全连接层, 7 神经网络与全连接层])

# 同样
out = F.avg_pool2d(x,2,stride = 2)
print(out.shape)
# torch.Size([1, 16, 7 神经网络与全连接层, 7 神经网络与全连接层])

#放大
out = F.interpolate(x,scale_factor=2,mode='nearest')
print(out.shape)
# torch.Size([1, 16, 28, 28])

out = F.interpolate(x,scale_factor=3,mode='nearest')
print(out.shape)
# torch.Size([1, 16, 42, 42])


#激活函数
layer = nn.ReLU(inplace = True)
out = layer(x)
print(out.shape)
# torch.Size([1, 16, 14, 14])



