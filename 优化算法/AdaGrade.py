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
    delta = 1e-7
    x = -20
    r = 0
    AdaGrad_x, AdaGrad_y = [], []
    for it in range(1000):
        AdaGrad_x.append(x), AdaGrad_y.append(ff.f(x))
        g = ff.df(x)
        r = r + g*g # 积累平方梯度
        x = x - lr /(delta + np.sqrt(r)) * g

    plt.xlim(-20, 20)
    plt.ylim(-2, 10)
    plt.plot(AdaGrad_x, AdaGrad_y, c="r", linestyle="-")
    plt.scatter(AdaGrad_x[-1],AdaGrad_y[-1],90,marker = "x",color="g")
    plt.title("AdaGrad,lr=%f"%(lr))
    plt.show()
    # plt.savefig("AdaGrad,lr=%f"%(lr) + ".png")
    plt.clf()