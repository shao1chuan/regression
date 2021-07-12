import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import matplotlib.pyplot
import time
import matplotlib.colors

learning_rate=0.01
epochs=50000
loss_array = []

# torch.manual_seed(12)
w1, b1 = torch.randn(16, 4, requires_grad=True),\
         torch.zeros(16, requires_grad=True)
w2, b2 = torch.randn(32, 16, requires_grad=True),\
         torch.zeros(32, requires_grad=True)
w3, b3 = torch.randn(16, 32, requires_grad=True),\
         torch.zeros(16, requires_grad=True)
w4, b4 = torch.randn(1, 16, requires_grad=True),\
         torch.zeros(1, requires_grad=True)



torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)
torch.nn.init.kaiming_normal_(w4)


# a1 = np.array([[6 随机梯度下降,8,4,3,10]])
# a2 = np.array([[4,9 卷积神经网络CNN,1,5,12]])
# a3 = np.array([[9 卷积神经网络CNN,5,2,3,14]])
# c = np.random.randint(1, high=100, size=(1000,5))
# c[:,-1] = (c[:,0]-c[:,2])*(c[:,1]-c[:,3])
# # print (c.shape)
# a = np.concatenate((a1,a2,a3,c))
# train_loader = torch.tensor(a).float()
# test_loader = torch.tensor(a4).float()
# print(train_loader.shape,target,test_loader)
x = torch.rand(1000, 4) * 10
y = (x[:, 0] - x[:, 2]) * (x[:, 1]-x[:, 3])


def forward(x):
    x = x@w1.t() + b1
    # x = F.sigmoid(x)
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    x = x @ w4.t() + b4
    # x = x.squeeze()
    return x
optimizer = optim.Adam([w1, b1, w2, b2, w3, b3, w4, b4], lr=learning_rate)
criteon = nn.MSELoss()

for epoch in range(epochs):
    target = y
    logits = forward(x)
    loss = criteon(logits, target)
    loss_array.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Train Epoch: {epoch} loss is {loss.item()}')

z = forward(torch.tensor([[6.0, 8.0, 4.0, 3.0]]))
print("result is ",z)
z = forward(torch.tensor([[4.0, 9.0, 1.0, 5.0]]))
print("result is ",z)
z = forward(torch.tensor([[9.0, 5.0, 2.0, 3.0]]))
print("result is ",z)
z = forward(torch.tensor([[6.0, 9.0, 3.0, 4.0]]))
print("result is ",z)

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(range(0, len(loss_array)), loss_array)
matplotlib.pyplot.show()


