import torch

a1 = torch.rand(4,32,8)
a2 = torch.rand(4,36,8)

a3 = torch.cat([a1,a2],dim = 1)
print(a3.shape)