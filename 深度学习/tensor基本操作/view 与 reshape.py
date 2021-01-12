import torch

a = torch.rand(4,1,28,29)

print(a.reshape(4*1*28,29).shape)

# 我也不知道是几行，反正是1列
print(a.reshape(-1,1).shape)

# 我也不知道是几列，反正是1行
print(a.reshape(1,-1).shape)

# 不分行列，改成1串
print(a.reshape(-1).shape)