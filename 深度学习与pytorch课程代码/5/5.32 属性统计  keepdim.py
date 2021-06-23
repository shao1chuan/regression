import torch

a1 = torch.randint(0,8,[4,2]).reshape(2,4).float()
print(a1)
# keepdim
print("notkeepdim ",a1.min(dim=1))
print("keepdim ",a1.min(dim=1,keepdim=True))
