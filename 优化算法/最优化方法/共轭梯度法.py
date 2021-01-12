# https://www.jb51.net/article/164529.htm

import random
import numpy as np
import matplotlib.pyplot as plt


def goldsteinsearch(f, df, d, x, alpham, rho, t):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d，t试探系数>1，
    '''
    flag = 0

    a = 0
    b = alpham
    fk = f(x)
    gk = df(x)

    phi0 = fk
    dphi0 = np.dot(gk, d)
    alpha = b * random.uniform(0, 1)

    while (flag == 0):
        newfk = f(x + alpha * d)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if (b < alpham):
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
    return alpha


def Wolfesearch(f, df, d, x, alpham, rho, t):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    σ∈(ρ,1)=0.75
    '''
    sigma = 0.75

    flag = 0

    a = 0
    b = alpham
    fk = f(x)
    gk = df(x)

    phi0 = fk
    dphi0 = np.dot(gk, d)
    alpha = b * random.uniform(0, 1)

    while (flag == 0):
        newfk = f(x + alpha * d)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            # if abs(np.dot(df(x + alpha * d),d))<=-sigma*dphi0:
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if (b < alpham):
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
    return alpha


def frcg(fun, gfun, x0):
    # x0是初始点，fun和gfun分别是目标函数和梯度
    # x,val分别是近似最优点和最优值，k是迭代次数
    # dk是搜索方向，gk是梯度方向
    # epsilon是预设精度，np.linalg.norm(gk)求取向量的二范数
    maxk = 5000
    rho = 0.6
    sigma = 0.4
    k = 0
    epsilon = 1e-5
    n = np.shape(x0)[0]
    itern = 0
    W = np.zeros((2, 20000))

    f = open("共轭.txt", 'w')

    while k < maxk:
        W[:, k] = x0
        gk = gfun(x0)
        itern += 1
        itern %= n
        if itern == 1:
            dk = -gk
        else:
            beta = 1.0 * np.dot(gk, gk) / np.dot(g0, g0)
            dk = -gk + beta * d0
            gd = np.dot(gk, dk)
            if gd >= 0.0:
                dk = -gk
        if np.linalg.norm(gk) < epsilon:
            break

        alpha = goldsteinsearch(fun, gfun, dk, x0, 1, 0.1, 2)
        # alpha=Wolfesearch(fun,gfun,dk,x0,1,0.1,2)
        x0 += alpha * dk

        f.write(str(k) + ' ' + str(np.linalg.norm(gk)) + "\n")
        print(k, alpha)
        g0 = gk
        d0 = dk
        k += 1

    W = W[:, 0:k + 1]  # 记录迭代点
    return [x0, fun(x0), k, W]


def fun(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def gfun(x):
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


if __name__ == "__main__":
    X1 = np.arange(-1.5, 1.5 + 0.05, 0.05)
    X2 = np.arange(-3.5, 4 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2  # 给定的函数
    plt.contour(x1, x2, f, 20)  # 画出函数的20条轮廓线

    x0 = np.array([-1.2, 1])
    x = frcg(fun, gfun, x0)
    print(x[0], x[2])
    # [1.00318532 1.00639618]
    W = x[3]
    # print(W[:, :])
    plt.plot(W[0, :], W[1, :], 'g*-')  # 画出迭代点收敛的轨迹
    plt.show()