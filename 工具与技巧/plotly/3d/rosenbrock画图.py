import plotly.graph_objects as go
import numpy as np
def rosenbrock(x1, x2):
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

# def aa(x1, x2):
#     return x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
x1 = np.arange(-5, 5, 0.5)
x2 =np.arange(-5, 5, 0.5)
x1,x2 = np.meshgrid(x1, x2)
fig = go.Figure(go.Surface(
    contours = {
        "x": {"show": True, "start": -4, "end": 4, "size": 0.4, "color":"white"},
"y": {"show": True, "start": -4, "end": 4, "size": 0.4, "color":"white"},
        "z": {"show": True, "start": 0.5, "end": 800, "size": 5}
    },
    x = x1,
    y = x2,
    z = rosenbrock(x1,x2)))

fig.show()