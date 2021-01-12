from visdom import Visdom

# 关于meshgrid
# https://blog.csdn.net/lllxxq141592654/article/details/81532855


import torch
import  torch.nn.functional as F
import numpy as np
from    matplotlib import pyplot as plt
def himmelblau(x):
    z = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    return z

def dawfig(f):
    x, y = np.arange(-60, 60, 0.1), np.arange(-60, 60, 0.1)
    print('x,y range:', x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print('X,Y maps:', X.shape, Y.shape)
    Z = f([X, Y])
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

viz = Visdom()
viz.line([0.],[0.],win='train_loss',opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))

# 1 设置开始点
x = torch.tensor([0.,0.],requires_grad=True)
# 2 设置优化器
optimizer = torch.optim.Adam([x],lr=1e-3)
# 3 循环找最优
for step in range(2000):
    # 3.1每次求函数值
    pred = himmelblau(x)
    # 3.2 优化器清零
    optimizer.zero_grad()
    # 3.3 方向向后传播
    pred.backward()
    # 3.4 向下走一步
    optimizer.step()
    #3.5 显示打印
    if step %20 == 0:
        print(f"step {step}: x = {x.tolist()},f(x) = {pred}")

    viz.line([pred.item()], [step], win='train_loss', update='append')

    viz.line([[pred.item(), 10000 / pred.item()]],
             [step], win='test', update='append')
    # viz.images(data.view(-1, 1, 28, 28), win='x')
    # viz.text(str(pred.detach().cpu().numpy()), win='pred',
    #          opts=dict(title='pred'))




