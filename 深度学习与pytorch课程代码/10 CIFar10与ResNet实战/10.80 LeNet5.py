import torch
from torch import nn
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.cnnunit = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        testdata = torch.randn(2, 3, 32, 32)
        out = self.cnnunit(testdata)
        # torch.Size([2, 16, 5, 5])
        print(out.shape)

        self.fullunit = nn.Sequential(
            nn.Linear(in_features=16*5*5,out_features=120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
        )
    def forward(self,x):
        # x = torch.Size([b, 3, 32, 32])
        out1 = self.cnnunit(x)
        # torch.Size([b, 16, 5, 5])
        banchsize = x.shape[0]
        out2 = out1.view(banchsize,-1)
        # torch.Size([b, 16*5*5])
        logits = self.fullunit(out2)
        # torch.Size([b, 10])
        return logits

def main():

    testdata = torch.randn(2, 3, 32, 32)
    test = Lenet5()
    print(test(testdata).shape)

if __name__ == '__main__':
    main()
