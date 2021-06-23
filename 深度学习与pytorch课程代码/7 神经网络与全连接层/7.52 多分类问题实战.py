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


# 2 参数初始化
w1, b1 = torch.randn(200, 784, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True),\
         torch.zeros(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

# 3 定义向前方法
def forward(x):

    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

# 4 定义优化器

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
# 定义损失函数
criteon = nn.CrossEntropyLoss()

# 5 训练神经网络
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        # 5.1 数据打平
        print(f"data: {data.shape}, target: {target.shape},target value:{target.data}")
        # data: torch.Size([200, 1, 28, 28]), target: torch.Size([200])
        data = data.view(-1, 28*28)
        # print("data:  ",data.shape)  200张  28*28 的图片 torch.Size([200, 784])
        # print("target:  ",target.shape) torch.Size([200])
        # 5.2 调用向前方法
        logits = forward(data)
        # print("logits:  ", logits.shape)
        # 5.3 定义损失函数
        loss = criteon(logits, target)
        # 5.4 清零，向后，下一步
        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()
        # 5.5 显示
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        test_loss += criteon(logits, target).item()
        print(f"logits: {logits.shape}, target: {target.shape}")
        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
