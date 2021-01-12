import torch

a = torch.tensor([2,3])
print(a)
a = torch.FloatTensor(2,3,4)
print("FloatTensor",a)
a = torch.rand(2,3)
print(a)
a = torch.rand_like(a)
print(a)
a = torch.randint(2,5,[2,3])
print("torch.randint(2,5,[2,3]) :",a)

# 正态分布 randn函数
a = torch.randn(3,3)
print(a)
a = torch.normal(mean = torch.full([10],1),std = torch.arange(1,0,-0.1))
print(a)

# full函数
a = torch.full([2,3],8)
print(a)

# linspace函数  0到1之间 创建10个数
a = torch.linspace(0,-1,10)
print("linspace:  ",a)

# logspace函数
a = torch.logspace(0,-1,10)
print(a)

# one/zeros/eye
a,b,c = torch.ones(2,3),torch.zeros(3,2),torch.eye(3,4)
print(a,b,c)

# randperm函数
a = torch.randperm(5)
print(a)

a= torch.from_numpy()