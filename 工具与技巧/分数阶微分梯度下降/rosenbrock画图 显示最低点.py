import plotly.graph_objects as go
import numpy as np
def rosenbrock(x1, x2):
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def plotF():
    x1 = np.arange(-5, 5, 0.5)
    x2 =np.arange(-5, 5, 0.5)
    x1,x2 = np.meshgrid(x1, x2)
    a1,a2 = np.array([3,4]),np.array([2.8,3.5])
    a3 = rosenbrock(a1,a1)
    fig = go.Figure(data = [
        go.Surface(
        contours = {        "x": {"show": True, "start": -4, "end": 4, "size": 0.4, "color":"white"}, "y": {"show": True, "start": -4, "end": 4, "size": 0.4, "color":"white"}, "z": {"show": True, "start": 0.5, "end": 800, "size": 5}    },
        x = x1,
        y = x2,
        z = rosenbrock(x1,x2),
        opacity=0.2),

        go.Scatter3d(
        x=a1, y=a1, z=a3,
        mode='markers',
        marker=dict(size=3,colorscale='Viridis',opacity=0.9),
        line=dict(width=6)
        ),


    ])

    fig.show()

if __name__ == '__main__':
    plotF()