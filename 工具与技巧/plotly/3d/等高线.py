# https://plotly.com/python/3d-surface-plots/
import plotly.graph_objects as go
import pandas as pd
import numpy as np

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)

fig = go.Figure(data=[go.Surface(contours = {
        "x": {"show": True, "start": 0, "end": 5, "size": 0.4, "color":"white"},
        "z": {"show": True, "start": -5, "end": 0, "size": 0.5}
    },z=Z, x=X, y=Y)])
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
fig.show()

