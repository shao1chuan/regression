import torch

a1 = torch.tensor([[1,2,3],[4,5,6]])
a2 = torch.tensor([[1,2,3],[4,5,5]])
print(a1.eq(a2))
print(a1>a2)