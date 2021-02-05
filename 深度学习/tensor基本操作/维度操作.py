import torch

# squeeze/unsqueeze
# transpose/t/permute
# expand/repeat

# view/reshape 只要整体size一致
a= torch.rand(4,1,28,28)
print(a.view(4,28,28))

# squeeze/unsqueeze  没有改变数据的大小，0 表示0维度之后插入
b = a.unsqueeze(1)
print(b.shape)
# torch.Size([4, 1, 1, 28, 28])
a =torch.rand(2,3)
print(a)
# tensor([[0.0989, 0.6838, 0.6195],
#         [0.2070, 0.0484, 0.6639]])
b = a.unsqueeze(-1)
print(b.shape)
# torch.Size([2, 3, 1])

