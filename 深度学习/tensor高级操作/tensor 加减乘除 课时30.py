# add = +

import numpy as np
import torch
# 相同位置元素加减乘除
a1 = np.array([[1,2],[3,4],[5,6]])
a2 = np.array([[1],[3],[5]])
a1 =torch.tensor(a1)
a2 =torch.tensor(a2)
print(f"a1+a2 = {a1+a2}\n a2.add(a1){a2.add(a1)}")
print(f"a1-a2 = {a1-a2}\n a2.sub(a1){a2.sub(a1)}")
print(f"a1*a2 = {a1*a2}\n a2.mul(a1){a2.mul(a1)}")
print(f"a1/a2 = {a1/a2}\n a2.div(a1){a2.div(a1)}")

# @ mutmul 相同  mm 只适合2D
a1 = np.array([[1,2],[3,4],[5,6]])
a2 = np.array([[1,2,3],[3,2,1]])
a1 =torch.tensor(a1)
a2 =torch.tensor(a2)
print(f"a1@a2 = {a1@a2}\n a2.matmul(a1){a2.matmul(a1)}")



