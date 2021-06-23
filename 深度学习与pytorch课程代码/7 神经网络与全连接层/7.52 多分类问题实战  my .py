import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


batch_size=200
learning_rate=0.01
epochs=10

# 1 数据装载
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

print(len(train_loader.dataset))

# 2 参数初始化
w1,b1 = torch.randn(200,784,requires_grad=True),torch.zeros(200,requires_grad=True)
w2,b2 = torch.randn(200,200,requires_grad=True),torch.zeros(200,requires_grad=True)
w3,b3 = torch.randn(10,200,requires_grad=True),torch.zeros(10,requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w1)
# 3 定义向前方法
def forward(x):
    x = x@w1.t()+b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x

# 4 定义优化器
optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3],lr = learning_rate)
# 定义损失函数
cri = nn.CrossEntropyLoss()

# 5 训练神经网络
for i in range(epochs):
    for idx, (data,target) in enumerate(train_loader):
        data = data.reshape(-1,28*28)
        logi = forward(data)
        loss = cri(logi,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f'Train Epoch: {i} [{len(data)}/{len(train_loader.dataset)}] \tLoss: {loss.item()}')

        test_loss = 0
        correct = 0









