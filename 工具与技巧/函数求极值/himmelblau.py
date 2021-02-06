import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def himmelblau(t):
	return (t[0] ** 2 + t[1] - 11) ** 2 + (t[0] + t[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
plt.show()

def plotf(losses):
    x = range(len(losses))
    y = losses
    plt.plot(x, y, label="losses")
    plt.legend(loc='best bbbbbbbbbb')
    plt.show()

import torch
x = torch.tensor([0., 0.], requires_grad = True)
optimizer = torch.optim.Adam([x,])
epochs = 20001
losses = []
for epoch in range(epochs):
    if epoch:
        optimizer.zero_grad()
        f.backward(retain_graph = True)
        optimizer.step()
    f = himmelblau(x)
    if epoch % 1000 == 0:
        print (f'epoch:{epoch} , x = {x.tolist()} , value = {f}')
    losses.append(f)
plotf(losses)