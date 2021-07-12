import torch
from torch import nn
from  torch.nn import functional as F

class ResBaseBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):

        super(ResBaseBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=3,stride=self.stride,padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out+x)
    
class ResDownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):

        super(ResDownBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=3,stride=self.stride[0],padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride[1],
                               padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(kernel_size=1,in_channels=self.in_channels,out_channels=self.out_channels,stride=self.stride[0]),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self,x):
        extrax = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out++extrax)



class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18,self).__init__()
        self.covn1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = nn.Sequential(
            ResBaseBlock(in_channels=64,out_channels=64,stride=1),
            ResBaseBlock(in_channels=64, out_channels=64, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResDownBlock(in_channels=64,out_channels=128,stride=[2,1]),
            ResBaseBlock(in_channels=128, out_channels=128, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResDownBlock(in_channels=128, out_channels=256, stride=[2,1]),
            ResBaseBlock(in_channels=256, out_channels=256, stride=1),
        )
        self.layer4 = nn.Sequential(
            ResDownBlock(in_channels=256, out_channels=512, stride=[2, 1]),
            ResBaseBlock(in_channels=512, out_channels=512, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,10)
    def forward(self,x):
        out = self.covn1(x)
        out = self.bn1(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(x.size(0),-1)
        out = self.fc(out)
        return out

def testresBaseBlock():
    testdata = torch.randn(10,64,56,56)
    resBaseBlock = ResBaseBlock(64,64,1)
    print(resBaseBlock(testdata).shape)

def testresDownBlock():
        # testdata = torch.randn(10, 64, 56, 56)
        # resDownBlock = ResDownBlock(64, 128, [2,1])
        # print(resDownBlock(testdata).shape)

        testdata = torch.randn(10,3,32,32)
        resnet18 = Resnet18()
        print(resnet18(testdata).shape)

if __name__ == '__main__':
    # testresBaseBlock()
    testresDownBlock()