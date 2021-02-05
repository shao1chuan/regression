import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data
import matplotlib.pyplot
import time
import matplotlib.colors

net_shape = [4, 16, 32, 16, 1]   # 每层节点数
#net_shape = [4, 16, 32, 16, 4]
net_f = [0, torch.relu, torch.relu, 0]   # 每层激活函数

W = []
B = []
Params = []
torch.manual_seed(12)
for n in range(0, len(net_shape)-1):
    w = torch.randn(net_shape[n], net_shape[n+1], requires_grad=True)
    W.append(w)
    b = torch.zeros(net_shape[n+1], requires_grad=True)
    B.append(b)
    Params.append(w)
    Params.append(b)


x = torch.rand(1000, 4) * 10
y = (x[:, 0] - x[:, 2]) * (x[:, 1]-x[:, 3])


print(f"x shape is {x.shape} y.shape is {y.shape}")

loss_array = []


def compute(data):
    global W, B

    buffer = data.clone()

    for i in range(0, len(net_shape)-1):
        buffer = torch.matmul(buffer, W[i]) + B[i]

        if net_f[i] != 0:
            buffer = net_f[i](buffer)

    result = buffer[:, 0]
    #result = torch.prod(buffer, dim=1)
    return result


loop = 50000


def train():
    global W, B, Params
    optimizer = torch.optim.Adam(Params, lr=0.01)

    for i in range(0, loop):

        r = compute(x)
        loss = (r - y).norm()
        loss_array.append(loss)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f"loss={loss} ")


t = time.time()
train()
print(time.time() - t)


for n in range(0, len(Params)):
    Params[n].requires_grad = False

z = compute(torch.tensor([[6.0, 8.0, 4.0, 3.0]]))
print(z)

z = compute(torch.tensor([[4.0, 9.0, 1.0, 5.0]]))
print(z)

z = compute(torch.tensor([[9.0, 5.0, 2.0, 3.0]]))
print(z)

z = compute(torch.tensor([[6.0, 9.0, 3.0, 4.0]]))
print(z)

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(range(0, len(loss_array)), loss_array)
matplotlib.pyplot.show()


