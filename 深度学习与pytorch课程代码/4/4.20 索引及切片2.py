import torch
from torch import tensor
# 4张图片，3通道，28*28像素
a1 = torch.rand(4,3,28,28)

# 4 表示第1个维度(图片个数)上的第1,3张图片
print(f" 1 表示第1个维度= {a1.index_select(0,torch.tensor([0,2])).shape}")
# 2 表示第3个维度上前8个像素
print(f"2 表示第3个维度上前8个像素 = {a1.index_select(2,torch.arange(8)).shape}")
# 3 表示所有维度所有数据
print(f"3 表示所有维度所有数据 = {a1[...].shape}")
# 4 表示第1个维度所有数据
print(f"5 表示所有维度所有数据 = {a1[0,...].shape}")
# 5 表示第2个维度所有数据
print(f"5 表示第2个维度所有数据 = {a1[:,1,...].shape}")

