import torch

a1 = torch.full([8],1.)
a2 = a1.reshape(2,4)
a3 = a1.reshape(2,2,2)
# 1范数
print(a1.norm(1),a2.norm(1),a3.norm(1))
# 2范数
print(a1.norm(2),a2.norm(2),a3.norm(2))

# 3范数 维度
print(a2.norm(1,dim = 1))
