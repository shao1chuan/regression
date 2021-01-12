import torch

a = torch.rand(4,3,28,28)
b = torch.rand(1,3,1,1)
# 只要是1的维数都可以expand，  -1表示维度不变
print(b.expand(1,3,28,28).shape)
print(b.expand(-1,-1,28,28).shape)
# repeat中参数表示复制的次数
print(a.repeat(1,2,1,1).shape)
# 将1，3 对换
print(a.transpose(1,3).shape)
# 对换之后换回去,  4,28,28,3 --- 4,28*28*3
aa = a.transpose(1,3).contiguous().view(4,28*28*3).view(4,28,28,3).transpose(1,3)
# 比较数据是否一致
print(torch.all(torch.eq(a,aa)))

print(aa.permute(0,1,2,3).shape,aa.permute(3,1,0,2).shape)

