import torch

a1 = torch.randint(0,8,[4,2]).reshape(2,4).float()
print(a1)
# prod累乘
print(a1.mean(),a1.prod())
# 返回索引,返回第几个元素
print(a1.argmax(),a1.argmin())
print(a1.argmax(dim = 0),a1.argmin(dim = 1))
