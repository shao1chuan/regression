import torch
# p1为实际已知的图片，狗为第一张图片
p1 = torch.tensor([1.,0.,0.,0.])
# a1 为图片估计概率
q1 = torch.tensor([0.4,0.3,0.05,0.2])
A1 = -(p1*torch.log2(q1)).sum()
print(A1)

q2 = torch.tensor([0.97,0.01,0.01,0.1])
A2 = -(p1*torch.log2(q2)).sum()
print(A2)