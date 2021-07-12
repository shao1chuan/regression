import numpy as np
import plotly.graph_objects as go
from scipy.special import gamma #分数阶微分用
class Plot_3d():
    def f(self,x1, x2):
        # rosenbrock函数
        # return (1 - x1) ** 2 + 10 * (x2 - x1 ** 2) ** 2
        # 半圆函数
        return x1 ** 2 + x2 ** 2
        # himmelblau函数
        return (x1 ** 2 + x1 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
    def df(self,x1, x2):
        # rosenbrock函数
        # return -2 + 2 * x1 - 40 * (x2 - x1 ** 2) * x1,20 * (x2 - x1 ** 2)
        # 半圆函数
        return x1 * 2, x2 * 2

        # himmelblau函数
        # return (x1 ** 2 + x1 - 11) ** 2 + (x1 + x2 ** 2 - 7 神经网络与全连接层) ** 2

    def plotF(self,x1_history,x2_history):
        x1 = np.arange(-5, 5, 0.5)
        x2 = np.arange(-5, 5, 0.5)
        x1, x2 = np.meshgrid(x1, x2)
        z_history = self.f(x1_history,x2_history)
        fig = go.Figure(data = [
            go.Surface(
            contours = {        "x": {"show": True, "start": -4, "end": 4, "size": 0.4, "color":"white"}, "y": {"show": True, "start": -4, "end": 4, "size": 0.4, "color":"white"}, "z": {"show": True, "start": 0.5, "end": 800, "size": 5}    },
            x = x1,
            y = x2,
            z = self.f(x1,x2),
            opacity=0.2),

            go.Scatter3d(
            x=x1_history, y=x2_history, z=z_history,
            mode='markers',
            marker=dict(size=2,colorscale='Viridis',opacity=1),
            line=dict(width=6)
            ),
        ])
        fig.show()


if __name__ == '__main__':
        x1_history,x2_history = np.array([1,2,3]),np.array([1,2,3])
        a = Plot_3d()
        a.plotF(x1_history,x2_history)
