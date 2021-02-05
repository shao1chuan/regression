import torch

a = torch.randn(2,3)

print(a.type(),type(a),isinstance(a,torch.FloatTensor))

b1 = torch.tensor(1.)
print(f"len(b1.shape) is {len(b1.shape)} len(b1.size()) is {len(b1.size())}")

b2 = torch.tensor(1.)
print(f"b2.shape is {b2.shape} b2.size() is {b2.size()}")
# shape 是属性，size()是方法

b3 = torch.tensor([1.])
print(f"b3.shape is {b3.shape} b3.size() is {b3.size()}")

b4 = torch.tensor([1.,2])
print(f"b4.shape is {b4.shape} b4.size() is {b4.size()}")

b5 = torch.tensor([[1.,2],[2,3]])
print(f"b5.shape is {b5.shape} b5.size() is {b5.size(0)}")

b6 = torch.tensor([[1.,2],[2,3]])
print(f"b5.shape is {b6.shape[0]} b5.size() is {b6.size(1)}")

a = torch.randn(2,3)
print("a的shape是：  "+str(a.shape))
print(f"a.size(0) is {a.size(0)}")
print(f"a.size(1) is {a.size(1)}")
print(f"a.shape[0] is {a.shape[0]}")
print(f"a.shape[1] is {a.shape[1]}")
print(f"a.dim() is {a.dim()}")
print(f"list(a.shape) is {list(a.shape)}")

# torch.FloatTensor <class 'torch.Tensor'> True
# len(b1.shape) is 0 len(b1.size()) is 0
# b2.shape is torch.Size([]) b2.size() is torch.Size([])
# b3.shape is torch.Size([1]) b3.size() is torch.Size([1])
# b4.shape is torch.Size([2]) b4.size() is torch.Size([2])
# b5.shape is torch.Size([2, 2]) b5.size() is 2
# b5.shape is 2 b5.size() is 2
# a的shape是：  torch.Size([2, 3])
# a.size(0) is 2
# a.size(1) is 3
# a.shape[0] is 2
# a.shape[1] is 3
# a.dim() is 2
# list(a.shape) is [2, 3]




