import numpy as np
import torch
import plotly.graph_objects as go
from plot_3d import Plot_3d

def main():
    epochs = 1000
    x1,x2 = torch.tensor([4.],requires_grad=True),torch.tensor([-4.],requires_grad=True)
    x1_history,x2_history,losses = [],[],[]
    optimizer = torch.optim.Adam([x1,x2], lr=0.1)
    p3 = Plot_3d()
    for epoch in range(epochs):
        loss = p3.f(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
        print(f"epoch is {epoch} ,x is {x1,x2}loss is {loss.item()}")
        x1_history.append(x1.clone().item())
        x2_history.append(x2.clone().item())

    z = p3.f(x1, x2)

    print("函数最小值是： ", z)
    p3.plotF(np.array(x1_history),np.array(x2_history))

if __name__ == '__main__':
    main()