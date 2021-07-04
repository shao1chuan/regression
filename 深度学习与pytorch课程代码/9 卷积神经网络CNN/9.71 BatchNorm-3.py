import  torch
import  torch.nn as nn
import  torch.nn.functional as F

x = torch.rand(1,16,7,7)
layer = nn.BatchNorm2d(16)
# 16 是  chanal
out = layer(x)
print(out.shape)
# torch.Size([1, 16, 7, 7])
print(vars(layer))




