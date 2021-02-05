import torch
import torch.nn as nn

x = torch.rand(1,16,14,14)
layer = nn.ReLU(inplace = True)
out = layer(x)
print(out.shape)
# torch.Size([1, 16, 14, 14])  小于0的都变成0