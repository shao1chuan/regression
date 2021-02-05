import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# https://blog.csdn.net/pengjian444/article/details/71075544
#
def plotf(loss):
    x = range(len(loss))
    plt.plot(x, loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


def cal_rosenbrock(x1, x2):
    """
    计算rosenbrock函数的值
    :param x1:
    :param x2:
    :return:
    """
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
def main():
    x1 = np.arange(-5, 5, 0.5)
    x2 = np.arange(-5, 5, 0.5)
    x1, x2 = np.meshgrid(x1, x2)
    Z = cal_rosenbrock(x1, x2)
    # g1 = 10 - x1 - 10 * x2
    # g2 = 10 * x1 - x2 - 10

    fig = go.Figure(data=[
        go.Surface(z=Z)
        # go.Surface(z=g1, showscale=False, opacity=0.9),
        # go.Surface(z=g2, showscale=False, opacity=0.9)

    ])

    fig.show()
if __name__ == '__main__':
    main()