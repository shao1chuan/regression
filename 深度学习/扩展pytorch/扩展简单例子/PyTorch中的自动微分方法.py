# https://blog.csdn.net/qq_39463274/article/details/105161769

import torch as t
x = t.ones(2,1, requires_grad=True)
print(f"初始x为： {x}  x的梯度为{x.grad}")
y = x + 2
z = t.mean(t.pow(y, 2))
z.backward()
print(f"x的梯度为： {x.grad}")   #这是求z对x的导数，求得tensor([3.])
print(f"y的梯度为： {y.grad}")   #y不是叶子节点没有梯度



x = t.randn((1),requires_grad=True)
print(x.grad)
#此时还为None
y = x + 2
y.backward()  #反向传播计算梯度
print(f"x的第一次梯度为： {x.grad}")
#此时，grad属性为tensor([1.])，梯度为１表明该函数沿着x轴正方向会增长的最快
y.backward()  #再来一次反向传播计算梯度
print(f"x的第二次梯度为： {x.grad}")
#此时，grad属性为tensor([２.])，表明梯度累加了，所以每次计算之前应当将梯度清零