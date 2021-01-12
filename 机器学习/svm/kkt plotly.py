# https://zhuanlan.zhihu.com/p/38163970
import numpy as np

import plotly.graph_objects as go


x1 = np.arange(-5,5,0.5)
x2 = np.arange(-5,5,0.5)
x1, x2 = np.meshgrid(x1, x2)
Z = x1**2 - 2*x1+1+x2**2+4*x2+4
g1 = 10-x1-10*x2
g2 = 10*x1-x2-10


fig = go.Figure(data=[
    go.Surface(z=Z),
    go.Surface(z=g1, showscale=False, opacity=0.9),
    go.Surface(z=g2, showscale=False, opacity=0.9)

])

fig.show()
