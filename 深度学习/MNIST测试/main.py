import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)



class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# C:\Users\dell\.virtualenvs\regression-J2dSGAi_\Scripts\python.exe D:/PycharmProjects/regression/code/深度学习/MNIST测试/main.py
# Train Epoch: 0 [0/60000 (0%)]	Loss: 2.313649
# Train Epoch: 0 [20000/60000 (33%)]	Loss: 1.936950
# Train Epoch: 0 [40000/60000 (67%)]	Loss: 1.175709
#
# Test set: Average loss: 0.0034, Accuracy: 8374.0/10000 (84%)
#
# Train Epoch: 1 [0/60000 (0%)]	Loss: 0.743421
# Train Epoch: 1 [20000/60000 (33%)]	Loss: 0.545286
# Train Epoch: 1 [40000/60000 (67%)]	Loss: 0.480503
#
# Test set: Average loss: 0.0020, Accuracy: 8912.0/10000 (89%)
#
# Train Epoch: 2 [0/60000 (0%)]	Loss: 0.387640
# Train Epoch: 2 [20000/60000 (33%)]	Loss: 0.391807
# Train Epoch: 2 [40000/60000 (67%)]	Loss: 0.277586
#
# Test set: Average loss: 0.0017, Accuracy: 9044.0/10000 (90%)
#
# Train Epoch: 3 [0/60000 (0%)]	Loss: 0.413547
# Train Epoch: 3 [20000/60000 (33%)]	Loss: 0.445348
# Train Epoch: 3 [40000/60000 (67%)]	Loss: 0.388933
#
# Test set: Average loss: 0.0015, Accuracy: 9117.0/10000 (91%)
#
# Train Epoch: 4 [0/60000 (0%)]	Loss: 0.366465
# Train Epoch: 4 [20000/60000 (33%)]	Loss: 0.319234
# Train Epoch: 4 [40000/60000 (67%)]	Loss: 0.281939
#
# Test set: Average loss: 0.0014, Accuracy: 9188.0/10000 (92%)
#
# Train Epoch: 5 [0/60000 (0%)]	Loss: 0.285491
# Train Epoch: 5 [20000/60000 (33%)]	Loss: 0.212992
# Train Epoch: 5 [40000/60000 (67%)]	Loss: 0.313581
#
# Test set: Average loss: 0.0013, Accuracy: 9237.0/10000 (92%)
#
# Train Epoch: 6 [0/60000 (0%)]	Loss: 0.271343
# Train Epoch: 6 [20000/60000 (33%)]	Loss: 0.285871
# Train Epoch: 6 [40000/60000 (67%)]	Loss: 0.274333
#
# Test set: Average loss: 0.0013, Accuracy: 9275.0/10000 (93%)
#
# Train Epoch: 7 [0/60000 (0%)]	Loss: 0.253012
# Train Epoch: 7 [20000/60000 (33%)]	Loss: 0.256780
# Train Epoch: 7 [40000/60000 (67%)]	Loss: 0.275217
#
# Test set: Average loss: 0.0012, Accuracy: 9312.0/10000 (93%)
#
# Train Epoch: 8 [0/60000 (0%)]	Loss: 0.242807
# Train Epoch: 8 [20000/60000 (33%)]	Loss: 0.296331
# Train Epoch: 8 [40000/60000 (67%)]	Loss: 0.239752
#
# Test set: Average loss: 0.0011, Accuracy: 9340.0/10000 (93%)
#
# Train Epoch: 9 [0/60000 (0%)]	Loss: 0.182446
# Train Epoch: 9 [20000/60000 (33%)]	Loss: 0.230142
# Train Epoch: 9 [40000/60000 (67%)]	Loss: 0.263742
#
# Test set: Average loss: 0.0011, Accuracy: 9365.0/10000 (94%)