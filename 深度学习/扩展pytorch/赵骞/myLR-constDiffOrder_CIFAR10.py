
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import gamma #分数阶微分用
##################### 自定义 #################################
torch.manual_seed(7)
diff_order=0.4
ofs = open("constDiffOrder_CIFAR10-result.txt","wt")
ofs.write("Diff-Order: %.2f\n\n" %(diff_order))
#############################################################

print(torch.__version__)
print("is cuda available? ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 1. Loading and normalizing CIFAR10
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../../../../data/cifar', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../../../../data/cifar', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#2. Define a Convolutional Neural Network
class F1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w,b):

        # x:[200,784],  w:[200,784], b[200,1]
        ctx.save_for_backward(x, w,b)
        # print(f"开始正向传播")
        x = x @ w.t()+b
        # X torch.Size([210, 200])
        return x

    @staticmethod
    def backward(ctx, grad_output):
        O, w, b = ctx.saved_tensors
        # O [200,784]  w[200,784]
        # error = np.dot(params['W3'].T, error) * self.sigmoid(params['X2'], derivative=True)
        # change_w['W2'] = np.outer(error, params['O1'])
        grad_x_input = grad_output @ w

        # grad_x_new [210,784]
        grad_w_input = grad_output.t() @ O
        grad_b_input = grad_output
        # grad_w_new [200,784]
        #######################   根据论文 （18）式 ####################################
        # grad_w_input = grad_w_input * (abs(w)**(1-diff_order))/gamma(2-diff_order)
        #grad_b_new = grad_b * (abs(b)**(1-diff_order))/gamma(2-diff_order)
        #################################################################################

        # print(f"开始反向传播 grad_x is {grad_x.shape}")
        return grad_x_input, grad_w_input, grad_b_input
w1, b1 = torch.randn(120, 16*5*5).to(device), torch.zeros(120).to(device)
w2, b2 = torch.randn(84, 120).to(device), torch.zeros(84).to(device)
w3, b3 = torch.randn(10, 84).to(device),  torch.zeros(10).to(device)


# w1, b1 = torch.randn(120, 16*5*5, requires_grad=True).to(device), torch.zeros(120, requires_grad=True).to(device)
# w2, b2 = torch.randn(84, 120, requires_grad=True).to(device), torch.zeros(84, requires_grad=True).to(device)
# w3, b3 = torch.randn(10, 84, requires_grad=True).to(device),  torch.zeros(10, requires_grad=True).to(device)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


w1.requires_grad = True
w2.requires_grad = True
w3.requires_grad = True
b1.requires_grad = True
b2.requires_grad = True
b3.requires_grad = True



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)

        x = F1.apply(x, w1, b1)
        x = F.relu(x)
        x = F1.apply(x, w2, b2)
        x = F.relu(x)
        x = F1.apply(x, w3, b3)
        return x

net = Net()
net = net.to(device)

# 3. Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
plist = list(net.parameters())
plist.append(w1)
plist.append(b1)
plist.append(w2)
plist.append(b2)
plist.append(w3)
plist.append(b3)
#optimizer = optim.SGD(plist, lr=0.001, momentum=0.9 卷积神经网络CNN)
optimizer = optim.SGD(plist, lr=0.001)


# 4. Train the network
#ofs = open("myLR-constDiffOrder-result.txt","wt")
#ofs.write("Diff-Order: %.2f\n\n" %(diff_order))
if __name__ == '__main__':
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs:[4,3,32,32], labels:[4]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                ofs.write('[%d, %5d] loss: %.3f\n' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


        #Let us look at how the network performs on the whole dataset.
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device),labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        ofs.write('Accuracy of the network on the 10000 test images: %d %%\n' % (100 * correct / total))

        #Hmmm, what are the classes that performed well, and the classes that did not perform well:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4): #这个4应该是指一个批次 4 个样本，前面在 load 数据集的时候用的
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            ofs.write('Accuracy of %5s : %2d %%\n\n' % (classes[i], 100 * class_correct[i] / class_total[i]))
ofs.close()