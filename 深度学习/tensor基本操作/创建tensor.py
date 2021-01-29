import torch
# 1 直接
a1 = torch.tensor([2,3])
print("torch.tensor([2,3])",a1)
a1 = torch.FloatTensor(2,3,4)
print("FloatTensor",a1)

# 2 rand
a2 = torch.rand(2,3)
print("torch.rand(2,3) ",a2)
a3 = torch.randint(2,5,[2,3])
print("torch.randint(2,5,[2,3]) :",a3)

# 2.1 like
a4 = torch.rand_like(a1)
print("a4 = torch.rand_like(a1) ",a4)

# 2.2 正态分布 randn函数
a5 = torch.randn(3,3)
print("a5 = torch.randn(3,3)",a5)
a6 = torch.normal(mean = torch.full([10],1),std = torch.arange(1,0,-0.1))
print("torch.normal(mean = torch.full([10],1),std = torch.arange(1,0,-0.1))",a6)

# 3 full函数
a7 = torch.full([2,3],8)
print("a7 = torch.full([2,3],8) ",a7)

# 4 linspace函数  0到1之间 创建10个数
a8 = torch.linspace(0,-1,10)
print("a8 linspace:  ",a8)

# logspace函数
a9 = torch.logspace(0,-1,10)
print("a9 = torch.logspace(0,-1,10)",a9)

# 5 one/zeros/eye
a,b,c = torch.ones(2,3),torch.zeros(3,2),torch.eye(3,4)
print(a,b,c)

# randperm函数
a10 = torch.randperm(5)
print("a10 = torch.randperm(5)",a10)
# 6
# a= torch.from_numpy()