import  torch
import  torch.nn as nn

layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=0)
x = torch.rand(1,1,28,28)

out = layer.forward(x)
print(out.shape)
# torch.Size([1, 3, 26, 26])

layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1)
out = layer.forward(x)
print(out.shape)
# torch.Size([1, 3, 28, 28])

layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=2,padding=1)
out = layer.forward(x)
print(out.shape)
# torch.Size([1, 3, 14, 14])

out = layer(x)
# 推荐这样使用，这样 有些hooks可以使用
