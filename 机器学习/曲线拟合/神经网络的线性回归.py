import  numpy as np
import  torch
import matplotlib.pyplot
import torch.nn.functional as F
loss_array = []
echo  =100
def forward(x):
    x = x@w1.double() + b1
    x = F.relu(x)
    x = x@w2.double() + b2
    x = F.relu(x)
    x = x@w3.double() + b3
    # x = F.relu(x)
    return x
points = np.genfromtxt("data.csv", delimiter=",")
pointx,pointy = (torch.from_numpy(points[:,0:1])),\
                torch.from_numpy(points[:,-1:])
w1,b1 = torch.randn(1,116, requires_grad=True),\
      torch.zeros(116,requires_grad=True)
w2,b2 = torch.randn(116,18, requires_grad=True),\
      torch.zeros(18,requires_grad=True)
w3,b3 = torch.randn(18,1, requires_grad=True),\
      torch.zeros(1,requires_grad=True)
optimizer = torch.optim.Adam([w1,w2,w3,b1,b2,b3], lr=0.001)
for i in range(0,echo):

 for x,y in zip(pointx,pointy):
    x = x.view(-1,1)
    y = y.view(-1, 1)
    # 1 forward
    out = forward(x)
    # 2 计算loss
    loss = torch.nn.functional.mse_loss(out, y)
    # 3 梯度清零
    optimizer.zero_grad()
    # 4 backward
    loss.backward()
    # 5 更新梯度
    optimizer.step()
    # 6 加入loss矩阵
    loss_array.append(loss)


matplotlib.pyplot.figure()
matplotlib.pyplot.plot(range(0, len(loss_array)), loss_array)
matplotlib.pyplot.show()

matplotlib.pyplot.figure()
# matplotlib.pyplot.plot(pointx, pointy)
matplotlib.pyplot.scatter(pointx, pointy)
z = []
for i in pointx:
    i = i.view(-1,1)
    z1 = forward(i).view(-1)
    z.append(z1)

matplotlib.pyplot.plot(pointx, z)
matplotlib.pyplot.show()
