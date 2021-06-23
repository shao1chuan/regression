import torch

a1 = torch.rand(4,4)
a2 = torch.ones(4,4)
a3 = torch.zeros(4,4)
print(a1)
print(torch.where(a1>0.5,a2,a3))