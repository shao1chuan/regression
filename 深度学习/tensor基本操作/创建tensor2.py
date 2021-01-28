import torch

a1 = torch.tensor([2,3,4,-4])
print(f"a1 = {a1}")

a2 = torch.tensor([-2,3,4,-4])
a2[a1>0] = 0
print(f"a2 = {a2}")

