import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter


class SignumActivation(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        output = input.sign().add(0.01).sign()
        return output, mean

    def backward(self, grad_output, grad_output_mean): #STE Part
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input=(2/torch.cosh(input))*(2/torch.cosh(input))*(grad_input)
        #grad_input[input.ge(1)] = 0 #great or equal
        #grad_input[input.le(-1)] = 0 #less or equal
        return grad_input
    
SignumActivation.apply
    
class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        #if input.size(1) != 784:
        #    input.data=input.data.sign().add(0.01).sign()
        if not hasattr(self.weight,'fp'):
            self.weight.fp=self.weight.data.clone()
        self.weight.data=self.weight.fp.sign().add(0.01).sign()
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.fp=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out


class BinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        #if input.size(1) != 3:
        #    input.data = input.data.sign().add(0.01).sign()
        if not hasattr(self.weight,'fp'):
            self.weight.fp=self.weight.data.clone()
        self.weight.data=self.weight.fp.sign().add(0.01).sign()
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.fp=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out
      

class Unit_BinarizedConvolution2D(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, ActivationLayer=1, BNaffine=False):
        super(Unit_BinarizedConvolution2D, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        if dropout!=0:
            self.dropout = nn.Dropout2d(dropout)
        self.ActivationLayer = ActivationLayer
        self.bn = nn.BatchNorm2d(output_channels, eps=1e-7, momentum=0.1, affine=BNaffine)
        if BNaffine==True:
            self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        self.conv=BinConv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)   #bias=False
    def forward(self, x):
        x = self.conv(x)
        if self.dropout_ratio!=0:
             x = self.dropout(x)
        x = self.bn(x)
        if self.ActivationLayer==1:
             x, mean = SignumActivation()(x)
        return x
    

class SignumActivationLayer(nn.Module):
    def forward(self, x):
        x, mean = SignumActivation()(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(
            Unit_BinarizedConvolution2D(3, 64, kernel_size=3, stride=1, padding=1, ActivationLayer=1,BNaffine=False),
            Unit_BinarizedConvolution2D(64, 64, kernel_size=3, stride=1, padding=1, ActivationLayer=1,BNaffine=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Unit_BinarizedConvolution2D(64, 128, kernel_size=3, stride=1, padding=1, ActivationLayer=1,BNaffine=False),
            Unit_BinarizedConvolution2D(128, 128, kernel_size=3, stride=1, padding=1, ActivationLayer=1,BNaffine=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            Unit_BinarizedConvolution2D(128, 256, kernel_size=3, stride=1, padding=1, ActivationLayer=1,BNaffine=False),
            Unit_BinarizedConvolution2D(256, 256, kernel_size=3, stride=1, padding=1, ActivationLayer=1,BNaffine=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=0),
            SignumActivationLayer()
        )
        
        self.classifier = nn.Sequential(
            BinarizeLinear(256, 10, bias=False),
            nn.BatchNorm1d(10, affine=False),
            #nn.Softmax(dim=1)
        )
        
        
    def forward(self, x):
        #for m in self.modules():
        #    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #        if hasattr(m.weight, 'data'):
        #            m.weight.data.clamp_(min=0.01)
        x = self.xnor(x)
        #x = x.view(x.size(0), 256)
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 256)
        x = self.classifier(x)
        return x

import time
import random
import matplotlib.pyplot as plt
import torch.utils.data as D
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms
import argparse

def timeSince(since):
    now = time.time()
    s = now - since
    #m = math.floor(s / 60)
    #s -= m * 60
    return s

def train(epoch):
    model.train()
    
    total_loss=0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward() #how can i modify this?
        
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1) #for test
        
        #for p in list(model.parameters()):
        #    if hasattr(p,'fp'):
        #        p.data.copy_(p.fp.clamp_(-1,1))
                
        for p in list(model.parameters()):
            if hasattr(p,'fp'):
                p.data.copy_(p.fp) 
         
        optimizer.step()
                       
        for p in list(model.parameters()):
            if hasattr(p,'fp'):
                p.fp.copy_(p.data.clamp_(-1,1))
                
        for p in list(model.parameters()):
            if hasattr(p,'fp'):
                p.data.copy_(p.fp.sign().add(0.01).sign())
                
        total_loss +=loss.item()
        
    total_loss /= len(train_loader)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]'
          .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader)))
    print('Train Loss: {}'.format(total_loss ))

def validate():
    global best_acc
    model.eval()
    test_total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(validate_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100. * correct / total #len(validate_loader)
    if acc > best_acc: #args.best_acc
        best_acc = acc
        #save_state(model, best_acc)
    test_total_loss /= len(validate_loader) 
       
    accur.append( 100.*correct/total)
    print('Validation Loss:',test_total_loss)
    print('Validation Accuracy:',acc)
    
    for param_group in optimizer.param_groups:
        print('Current Learning Rate:', param_group['lr'])
        
    print('Best Accuracy:: ',best_acc)
    print('--------------------------------------------')

set_batch_size=100

#torch.cuda.manual_seed(1)
torch.cuda.seed()
    #train_loader
train_loader = D.DataLoader(datasets.CIFAR10('./data', train=True, download=True,
                                             transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                           transforms.RandomAffine(degrees=0, translate=(.1,.1), scale=None, shear=None, resample=False, fillcolor=0),
                                                                           transforms.ToTensor()
                                                                          ]))
                            ,batch_size=set_batch_size, shuffle=True) #500->args.test_batch_size
                                                
    
    #test_loaer
validate_loader = D.DataLoader(datasets.CIFAR10('./data', train=False, 
                                            transform=transforms.Compose([transforms.ToTensor() #transforms.ToPILImage()
                                                                         ]))
                           ,batch_size=set_batch_size, shuffle=False)

model=Net()
model.cuda() #to GPU

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=0.0001) 
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.8) 

#lambda1 = lambda epoch: 0.95 ** epoch
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

scheduler= optim.lr_scheduler.MultiStepLR(optimizer, 
                                          milestones=[300,400,500,550,600,620,640,660,680,700], 
                                          gamma=0.1, last_epoch=-1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #cifar10
start = time.time()
time_graph=[]
e=[]
accur=[]
tlos=[]
best_acc=0

last_epoch=720

for epoch in range(1, last_epoch): 
    e.append(epoch)
    train(epoch)   
    seco=timeSince(start)
    time_graph.append(seco)
    validate()
    scheduler.step()

print(time_graph)
plt.title('Training for CIFAR10 with epoch', fontsize=20)
plt.ylabel('time (s)')
plt.plot(e,time_graph)
plt.show()
plt.title('Accuracy With epoch', fontsize=20)
plt.plot(e,accur)
plt.show()
plt.title('Test loss With epoch', fontsize=20)
plt.plot(tlos)
plt.show()