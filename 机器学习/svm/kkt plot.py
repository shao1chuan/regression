# https://zhuanlan.zhihu.com/p/38163970
import numpy as np
from matplotlib import pyplot as plt
fig = plt.figure()  #定义新的三维坐标轴
ax1 = plt.axes(projection='3d')

x1 = np.arange(-5,5,0.5)
x2 = np.arange(-5,5,0.5)
x1, x2 = np.meshgrid(x1, x2)
Z = x1**2 - 2*x1+1+x2**2+4*x2+4
g1 = 10-x1-10*x2
g2 = 10*x1-x2-10




ax1.plot_surface(x1,x2,Z,cmap='rainbow')
ax1.plot_surface(x1,x2,g1,cmap='rainbow')
ax1.plot_surface(x1,x2,g2,cmap='rainbow')
ax1.contour(x1,x2,Z, zdim='z',offset=-2,cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
plt.show()
