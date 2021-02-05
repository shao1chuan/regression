import numpy as np
import torch
import plotly.graph_objects as go
# https://blog.csdn.net/pengjian444/article/details/71075544
def plotf():
    x1 = np.arange(-5, 5, 0.5)
    x2 = np.arange(-5, 5, 0.5)
    x1, x2 = np.meshgrid(x1, x2)
    Z = rosenbrock(x1, x2)
    # g1 = 10 - x1 - 10 * x2
    # g2 = 10 * x1 - x2 - 10

    fig = go.Figure(data=[
        go.Surface(z=Z)
        # go.Surface(z=g1, showscale=False, opacity=0.9),
        # go.Surface(z=g2, showscale=False, opacity=0.9)

    ])

    fig.show()
def rosenbrock(x1, x2):
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def main():
    epochs = 50000
    x1,x2 = torch.tensor([15.],requires_grad=True),torch.tensor([15.],requires_grad=True)

    losses = []
    optimizer = torch.optim.Adam([x1,x2], lr=0.01)

    for epoch in range(epochs):
        loss = rosenbrock(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
        print(f"epoch is {epoch} loss is {loss.item()}")

    y = rosenbrock(x1, x2)
    print("函数最小值是： ", y)
    plotf()

if __name__ == '__main__':
    main()