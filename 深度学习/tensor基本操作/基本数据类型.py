import torch

a = torch.randn(2,3)

print(a.type(),type(a),isinstance(a,torch.FloatTensor))

b = torch.tensor(1.)
print(len(b.shape),len(b.size()))
# shape 是属性，size()是方法

a = torch.randn(2,3)
print("a的shape是：  "+str(a.shape))

print(a.size(0),a.size(1),a.shape[0],a.shape[1],a.dim(),list(a.shape))
