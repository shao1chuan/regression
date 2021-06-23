import torch

a1 = torch.tensor([0.25,0.25,0.25,0.25])
# 求a1 交叉熵
a11 = -(a1*torch.log2(a1)).sum()
print(a11)

a1 = torch.tensor([0.01,0.01,0.02,0.96])
# 求a1 交叉熵
a11 = -(a1*torch.log2(a1)).sum()
print(a11)