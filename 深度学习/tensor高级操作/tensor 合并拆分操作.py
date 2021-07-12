import torch
a1 = torch.rand(3,3,28,28)
a2 = torch.rand(3,3,28,28)
a3 = torch.rand(6,3,28,28)

a = torch.cat([a1,a2,a3],dim = 0)
print(f"a shape is {a.shape}")
# a shape is torch.Size([12, 3, 28, 28])

b = torch.cat([a1,a2],dim = 1)
print(f"b shape is {b.shape}")
# b shape is torch.Size([3, 6 随机梯度下降, 28, 28])

a1 = torch.rand(1,2)
a2 = torch.rand(1,2)
c = torch.stack([a1,a2],dim=2)
print(f"c shape is {c.shape}")
# c shape is torch.Size([1, 2, 2])

d = torch.stack([a1,a2,a1,a2],dim=0)
print(f"d shape is {d.shape}")
# d shape is torch.Size([4, 1, 2])

d1,d2,d3,d4 = d.split(1,dim = 0)
# 根据相同长度拆分
print(f"d1,d2 shape is {d1.shape,d2.shape}")
# d1,d2 shape is (torch.Size([1, 1, 2]), torch.Size([1, 1, 2]))

d3,d4 = d.split([3,1],dim = 0)
# 根据不同长度拆分
print(f"d3,d4 shape is {d3.shape,d4.shape}")
# d3,d4 shape is (torch.Size([3, 1, 2]), torch.Size([1, 1, 2]))

d5,d6 = d.chunk(2,dim = 0)
# 根据数量拆分，才分2个
print(f"d5,d6 shape is {d5.shape,d6.shape}")
# d5,d6 shape is (torch.Size([2, 1, 2]), torch.Size([2, 1, 2]))
