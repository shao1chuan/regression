# Adam
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from AA import Function
ff = Function()


for i in range(15):
    # 绘制原来的函数
    plt.plot(ff.points_x, ff.points_y, c="b", alpha=0.5, linestyle="-")
    # 算法开始
    lr = pow(1.5,-i)*32
    delta = 1e-6
    rou = 0.8
    x = -20
    r = 0
    RMSProp_x, RMSProp_y = [], []
    for it in range(1000):
        RMSProp_x.append(x), RMSProp_y.append(ff.f(x))
        g = ff.df(x)
        r = rou * r + (1-rou)*g*g # 积累平方梯度
        x = x - lr /(delta + np.sqrt(r)) * g

    plt.xlim(-20, 20)
    plt.ylim(-2, 10)
    plt.plot(RMSProp_x, RMSProp_y, c="r", linestyle="-")
    plt.scatter(RMSProp_x[-1],RMSProp_y[-1],90,marker = "x",color="g")
    plt.title("RMSProp,lr=%f,rou=%f"%(lr,rou))
    # plt.savefig("RMSProp,lr=%f,rou=%f"%(lr,rou) + ".png")
    plt.show()
    plt.clf()