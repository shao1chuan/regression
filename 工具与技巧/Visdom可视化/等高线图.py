# https://zhuanlan.zhihu.com/p/32025746
import visdom
import numpy as np
viz = visdom.Visdom()


# contour
x1 = np.arange(-50,50,0.5)
x2 = np.arange(-50,50,0.5)
x1, x2 = np.meshgrid(x1, x2)
Z = x1**2 - 2*x1+1+x2**2+4*x2+4
g1 = 10-x1-10*x2
g2 = 10*x1-x2-10

viz.contour(X=Z, win = 'surface',opts=dict(colormap='Viridis'))
# surface
# viz.surf(X=Z,win = 'surface',opts=dict(
#         xtickmin=-50,
#         xtickmax=50,
#
#         ytickmin=-50,
#         ytickmax=50,
#
#         markersymbol='cross-thin-open',))