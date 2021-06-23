import torch

a1 = torch.tensor([[2.,3.],[3.,3.]])
a2 = torch.ones(2,3)
a3 = a1@a2
print(a3)

a4 = torch.rand(4,284)
a5 = torch.rand(512,284)
a6 = a4@a5.t()
print(a6.shape)