import torch
from torch import nn
from ResBlk import ResBlk

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        stride,ch_in, ch_out,outclass = 2,3,16,10
        #[b,3,224,224]
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            # [2, 16, 224, 224]
            # nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

        )
        # [2, 16, 224, 224]
        self.blk1 = ResBlk(ch_out,ch_out,stride)
        # [2, 16, 224, 224]
        self.blk2 = ResBlk(ch_out, ch_out*2,stride)
        # [2, 32, 224, 224]
        self.outlayer = nn.Linear(32*224*224,outclass)

    def forward(self,x):
        x = self.conv1(x)
        print("after conv1:", x.shape)
        x = self.blk2(self.blk1(x))
        print("after blk2:", x.shape)
        x = x.view(x.size(0),-1)
        out = self.outlayer(x)
        return out
def main():
    data = torch.randn(2,3,224,224)
    model = ResNet18()
    print(model(data).shape)
if __name__ == '__main__':
    main()