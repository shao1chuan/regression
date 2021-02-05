import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


learning_rate=0.1
epochs=800

torch.manual_seed(1)
w1, b1 = torch.randn(3, 4, requires_grad=True),\
         torch.zeros(3, requires_grad=True)
w2, b2 = torch.randn(3, 3, requires_grad=True),\
         torch.zeros(3, requires_grad=True)


w3, b3 = torch.randn(1, 3, requires_grad=True),\
         torch.zeros(1, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

a1 = [6,8,4,3,10]
a2 = [4,9,1,5,12]
a3 = [9,5,2,3,14]
a4 = [6,9,3,4]

a = [a1,a2,a3]
train_loader = torch.tensor(a).float()

test_loader = torch.tensor(a4).float()

# print(train_loader.shape,target,test_loader)


def forward(x):
    x = x@w1.t() + b1
    # x = F.relu(x)
    x = x@w2.t() + b2
    # x = F.relu(x)
    x = x @ w3.t() + b3

    return x



optimizer = optim.Adam([w1, b1, w2, b2,w3, b3], lr=learning_rate)
criteon = nn.MSELoss()

for epoch in range(epochs):

    for idx,data in enumerate(train_loader):
        target = data[-1]
        tmp = data[0:4]
        logits = forward(tmp)
        # print(logits,target)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} loss is {loss.item()}')

result  = forward(test_loader)
print("result is ",result)



