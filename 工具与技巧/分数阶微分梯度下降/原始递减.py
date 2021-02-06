

from plot_3d import Plot_3d
import numpy as np

def minf():
    x1,x2 = 4.,4.
    z_history = []
    x1_history = []
    x2_history = []
    epochs = 1500
    lr = 0.001
    p3 = Plot_3d()
    for epoch in range(epochs):
        z = p3.f(x1, x2)
        x1_history.append(x1)
        x2_history.append(x2)
        z_history.append(z)
        dx1,dx2 = p3.df(x1, x2)
        x1 -= lr*dx1
        x2 -= lr*dx2
        print("epoch ", epoch, "z:", z)
    p3.plotF(np.array(x1_history),np.array(x2_history))
    return z

if __name__ == '__main__':
    w = minf()
    print(w)