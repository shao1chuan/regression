import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
import math
import numpy as np


learning_rate=0.1
epochs=800

torch.manual_seed(12)
w1 = torch.ones(2, 4, requires_grad=True)
w2 = torch.ones(1, 2, requires_grad=True)
'''
print("w1.type()",w1)
w1[0,0]=1
w1[0,1]=0
w1[0,2]=-1
w1[0,3]=0
w1[1,0]=0
w1[1,1]=1
w1[1,1]=0
w1[1,1]=-1
w2[0,0]=1
w2[0,1]=1
print("w1.type()",w1)
'''




#w3, b3 = torch.randn(1, 3, requires_grad=True),\
#         torch.zeros(1, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
#torch.nn.init.kaiming_normal_(w3)

a1 = [6,8,4,3,10]
a2 = [4,9,1,5,12]
a3 = [9,5,2,3,14]
a4 = [4,3,2,1,4]
a5 = [5,4,3,2,4]
a6 = [5,3,2,1,6]


a7 = [6,9,3,4]

a = [a1,a2,a3,a4,a5,a6]
train_loader = torch.tensor(a).float()

test_loader = torch.tensor(a7).float()

# print(train_loader.shape,target,test_loader)


def forward(x):
    x = x@w1.t() #+ b1
    print("x1=",x)
    x = F.relu(x)+0.1
    #print("x2=",x)
    x = torch.log(x)
    #print("x3=",x)
    x = x@w2.t()# + b2
    #print("x4=",x)
    #x = F.relu(x)
    x = torch.exp(x)
    #print("x5=",x)
    #x = x @ w3.t() + b3
    print(w1.grad)

    return x



optimizer = optim.Adam([w1, w2], lr=learning_rate)
criteon = nn.MSELoss()

for epoch in range(epochs):
    loss=0
    for idx,data in enumerate(train_loader):
        target = data[-1]
        print("target=",target)
        tmp = data[0:4]
        #print("tmp=",tmp)
        #print("w1=",w1)
        #print("w2=",w2)
        logits = forward(tmp)
        #print("logits=",logits)
        # print(logits,target)
        loss += criteon(logits, target)
        #print("loss=",loss)
    #print("w1:",w1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Train Epoch: {epoch} loss is {loss.item()}')


result1 = forward(train_loader[0][0:4])
result2 = forward(train_loader[1][0:4])
result3 = forward(train_loader[2][0:4])
result4  = forward(test_loader)
print("results is ",result1.item(), result2.item(), result3.item(), result4.item())

print("w1:",w1)




