import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def H(x, y):
    return np.matrix([[1200 * x * x - 400 * y + 2, -400 * x],
                      [-400 * x, 200]])


def grad(x, y):
    return np.matrix([[2 * x - 2 + 400 * x * (x * x - y)],
                      [200 * (y - x * x)]])


def delta_newton(x, y):
    alpha = 1.0
    delta = alpha * H(x, y).I * grad(x, y)
    return delta


# ----- 绘制等高线 -----
# 数据数目
n = 256
# 定义x, y
x = np.linspace(-1, 1.1, n)
y = np.linspace(-1, 1.1, n)

# 生成网格数据
X, Y = np.meshgrid(x, y)

plt.figure()
# 填充等高线的颜色, 8是等高线分为几部分
plt.contourf(X, Y, f(X, Y), 5, alpha=0, cmap=plt.cm.hot)
# 绘制等高线
C = plt.contour(X, Y, f(X, Y), 8, locator=ticker.LogLocator(), colors='black', linewidth=0.01)
# 绘制等高线数据
plt.clabel(C, inline=True, fontsize=10)
# ---------------------

x = np.matrix([[-0.3],
               [0.4]])

tol = 0.00001
xv = [x[0, 0]]
yv = [x[1, 0]]

plt.plot(x[0, 0], x[1, 0], marker='o')

for t in range(100):
    delta = delta_newton(x[0, 0], x[1, 0])
    if abs(delta[0, 0]) < tol and abs(delta[1, 0]) < tol:
        break
    x = x - delta
    xv.append(x[0, 0])
    yv.append(x[1, 0])

plt.plot(xv, yv, label='track')
# plt.plot(xv, yv, label='track', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function')
plt.legend()
plt.show()