import torch
# 1 按照长度split
a1 = torch.rand(4,32,8)
b1,b2 = a1.split([3,1],dim= 0)

print(b1.shape,b2.shape)

# 1 按照数量split

b1,b2,b3,b4 = a1.split(1,dim= 0)

print(b1.shape,b2.shape)