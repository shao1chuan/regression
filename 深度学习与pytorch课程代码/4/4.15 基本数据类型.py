import torch
from torch import tensor

a1 = torch.randn(2,3)
print(a1,a1.type())
print(isinstance(a1,torch.cuda.FloatTensor))
a1 = a1.cuda()
print(isinstance(a1,torch.cuda.FloatTensor))

a2 = torch.tensor(1.)
print(a2,a2.shape,a2.size())
