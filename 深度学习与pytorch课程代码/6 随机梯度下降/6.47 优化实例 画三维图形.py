# 关于meshgrid
# https://blog.csdn.net/lllxxq141592654/article/details/81532855


import torch
import  torch.nn.functional as F
import numpy as np
from    matplotlib import pyplot as plt
def himmelblau(x):
    z = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    return z

x,y = np.arange(-60,60,0.1),np.arange(-60,60,0.1)
print('x,y range:', x.shape, y.shape)
X,Y = np.meshgrid(x,y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

