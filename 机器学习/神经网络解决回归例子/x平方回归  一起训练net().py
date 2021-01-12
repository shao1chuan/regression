import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import matplotlib.pyplot
# 装载数据
x = torch.linspace(-2., 2., 1000)
y = x ** 2
x = x.unsqueeze(-1)
y = y.unsqueeze(-1)
z = torch.zeros(1000)
loss_array = []
epoch =  5000

# 定义神经网络层结构
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

net = Net()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for i in range(epoch):
    out = net(x)
    loss = F.mse_loss(out, y)
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
matplotlib.pyplot.plot(x.reshape(-1),y.reshape(-1))
z= net(x).reshape(-1)
matplotlib.pyplot.plot(x.reshape(-1), z.detach().numpy())

matplotlib.pyplot.show()


