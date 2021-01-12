import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim

import matplotlib.pyplot
epochs = 100
learning_rate=0.01
x = torch.linspace(-2., 2., 1000)
y = x**2

loss_array = []

w1, b1 = torch.randn(1, 16, requires_grad=True),\
         torch.zeros(16, requires_grad=True)
w2, b2 = torch.randn(16, 8, requires_grad=True),\
         torch.zeros(8, requires_grad=True)
w3, b3 = torch.randn(8, 1, requires_grad=True),\
         torch.zeros(1, requires_grad=True)

def forward(x):
    x = x@w1 + b1
    x = F.relu(x)
    x = x@w2 + b2
    x = F.relu(x)
    x = x@w3 + b3
    # x = F.relu(x)
    return x

optimizer = optim.Adam([w1, b1, w2, b2, w3, b3], lr=learning_rate)
# criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for xi in x:
        xi = xi.view(-1, 1)
        out = forward(xi)
        yi = xi**2
        # 1计算loss
        loss = F.mse_loss(out, yi)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()
        loss_array.append(loss)
        # print(w1,w2,w3)

    print(f"loss={loss}")

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(range(0, len(loss_array)), loss_array)
matplotlib.pyplot.show()


x = torch.linspace(-2., 2., 1000)
y = x**2
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(x, y)
z = []
for i in x:
    i = i.view(-1,1)
    z1 = forward(i).view(-1)
    z.append(z1)

matplotlib.pyplot.plot(x, z)

matplotlib.pyplot.show()

