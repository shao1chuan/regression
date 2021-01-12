import torch

a = torch.rand(4,3,28,28)
print(a[:2,:1,:,:].shape)
print(a[1:2,-1:,-1:,-1:])

# 对图片间隔采样
print(a[::2,:,:,:].shape)
# 对图片隔条采样
print(a[:,:,0:28:2,0:28:2].shape)
