import torch
from torch import nn
import torch.nn.functional as F
from ResBlk import ResBlk
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(64)
        )
        self.blk1 =ResBlk(64,128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)
        self.outlayer = nn.Linear(512,10)


    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        # print('after conv:', out.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 4, 4]
        out = F.adaptive_avg_pool2d(out, [1, 1])
        # print('after pool:', x.shape)
        out = out.view(x.size(0), -1)
        out = self.outlayer(out)
        return out
def main():
    b = 100
    x = torch.randn(2,3,32,32)
    model = ResNet18()
    out = model(x)
    print(out)
if __name__ == '__main__':
    main()



