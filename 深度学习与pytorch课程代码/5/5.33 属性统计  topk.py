import torch

a1 = torch.rand(4,10)
print(a1,a1.max(dim = 1,keepdim=True))
a2 = a1.topk(2)
print(f"topk is {a2}")
a3 = a1.kthvalue(2,dim = 1)
print(f"第2小的数kthvalue is {a3}")