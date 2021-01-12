import torch

a1 = torch.rand(2,3)
aa = torch.rand([4,3,28,28])
a = torch.rand([1,3,1,1])

# unsqueeze从-1的后面插入一个维度,也就是说插入1列，
# 0是从前面插入一个维度，也就是说插入1行
print(a1,a1.shape)
print("a1.unsqueeze(0),a1.unsqueeze(-1):  ",a1.unsqueeze(0).shape,a1.unsqueeze(-1).shape)
print(a1.unsqueeze(0).shape,a1.unsqueeze(-1).shape)

# squeeze把 数值维度挤压掉
print(a.squeeze(2).shape,a.squeeze(1).shape,a.squeeze().shape)
