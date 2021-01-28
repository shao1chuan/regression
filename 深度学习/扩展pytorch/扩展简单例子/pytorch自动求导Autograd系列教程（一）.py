# https://blog.csdn.net/qq_27825451/article/details/89393332
# 例子1
import torch
x = torch.tensor(3.0, requires_grad=True)
y = torch.pow(x, 2)

# 判断x,y是否是可以求导的
print(x.requires_grad)
print(y.requires_grad)

# 求导，通过backward函数来实现
y.backward()

# 查看导数，也即所谓的梯度
print(f"例子1 x.grad is {x.grad}")
# 只有当所有的“叶子变量”，即所谓的leaf variable都是不可求导的，那函数y才是不能求导
print(f"例子1 y.grad is {y.grad}")

# 例子2

# 创建一个二元函数，即z=f(x,y)=x2+y2，x可求导，y设置不可求导
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=False)
z = torch.pow(x, 2) + torch.pow(y, 2)

# 判断x,y是否是可以求导的
print(x.requires_grad)
print(y.requires_grad)
print(z.requires_grad)

# 求导，通过backward函数来实现
z.backward()

# 查看导数，也即所谓的梯度
print(f"例子2 x.grad is {x.grad}")
print(f"例子2 y.grad is {y.grad}")

'''运行结果为：
True       # x是可导的
False      # y是不可导的
True       # z是可导的，因为它有一个 leaf variable 是可导的，即x可导
tensor(6.) # x的导数
None       # 因为y不可导，所以是none
'''
