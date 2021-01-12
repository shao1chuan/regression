import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import matplotlib.pyplot
# 装载数据
x = torch.linspace(-2., 2., 1000)
y = x ** 2
loss_array = []
epoch =  50

# 定义神经网络层结构
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        # self.fc1 = nn.Linear(1, 16)
        # self.fc2 = nn.Linear(16, 8)
        # self.fc3 = nn.Linear(8, 1)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8, 1),
        )

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    def forward(self, x):
        x = self.model(x)
        return x

net = Net()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for i in range(epoch):
 for xi in x:
    xi = xi.view(-1,1)
    yi = xi ** 2
    out = net(xi)
    loss = F.mse_loss(out, yi)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_array.append(loss.item())
 if i % 10 == 0:
     print(f"loss={loss} time={i}")

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(range(0, len(loss_array)), loss_array)
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(x, y)
z = []
for xi in x:
    xi = xi.view(-1,1)
    z1 = net(xi).view(-1)
    z.append(z1)

matplotlib.pyplot.plot(x, z)

matplotlib.pyplot.show()


