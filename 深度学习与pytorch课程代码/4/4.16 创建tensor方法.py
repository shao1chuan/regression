import torch
from torch import tensor
import numpy as np

# 4 直接从list中创建
a1 = torch.tensor([1, 2, 3])
print("4 直接从list中创建", a1, "\n")
# 2 从numpy中创建
a1 = np.array([1., 2., 300])
a2 = torch.from_numpy(a1)
print("2 从numpy中创建", a1, a2, "\n")
# 3 定义矩阵
a1 = np.ones([2, 3])
a2 = torch.from_numpy(a1)
print('3 定义矩阵', a1, a2, "\n")

# 4 用大写Tensor创建，参数是shape，2行 3列矩阵
a1 = torch.Tensor(2, 3)
a2 = torch.IntTensor(2, 3)
print(f'4 用大写Tensor创建，参数是shape   a1 = {a1} \n a2 = {a2} \n')
a2 = torch.FloatTensor([2, 3])
print('5 用大写Tensor创建,这种方式不建议使用', a2, "\n")
