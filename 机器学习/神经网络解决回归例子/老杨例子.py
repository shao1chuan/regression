import torch
import torch.nn
import torch.nn.functional
import torch.optim
import numpy
import matplotlib
import matplotlib.pyplot


x = torch.linspace(-2., 2., 1000)
y = x ** 2

width = 8
depth = 4


w = torch.rand(depth, width, width)
b = torch.rand(depth,  width)

print(f"w:{w[0]}")
loss_array = []


def compute(data):
    global w, width, depth, b

    buffer = data.unsqueeze(1)
    buffer = buffer.repeat(1, width)
    buffer[:, 1:] = 0

    for i in range(0, depth):
        buffer = torch.matmul(buffer, w[i]) + b[i]

        if 0 < i < depth-1:
            buffer = torch.sigmoid(buffer)

    result = buffer[:, 0] #.sum(1) / width
    return result


loop = 5000
lr_base = 0.01

w.requires_grad = True
b.requires_grad = True


def train():
    global w, b

    for i in range(0, loop):

        r = compute(x)
        #loss = torch.nn.functional.mse_loss(r, y)
        loss = (r - y).norm()
        #loss = (r - y).abs().sum()
        loss_array.append(loss)

        loss.backward()

        lr = lr_base / (1 + 0.001 * i)

        w.data = w - lr * w.grad.data
        b.data = b - lr * b.grad.data

        w.grad.zero_()
        b.grad.zero_()
        #optimizer.step()

        if i % 100 == 0:
            print(f"loss={loss}")

    print(f"train finish! {loss}")


train()
print(f"w:{w[0]}")
w.requires_grad = False
b.requires_grad = False
z = compute(x)

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(range(0, len(loss_array)), loss_array)
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(x, y)
matplotlib.pyplot.plot(x, z)

matplotlib.pyplot.show()

