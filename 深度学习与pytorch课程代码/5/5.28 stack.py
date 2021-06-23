import torch

a1 = torch.rand(4,32,8)
a2 = torch.rand(4,32,8)

a3 = torch.stack([a1,a2],dim = 1)
print(a3.shape)