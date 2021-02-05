import torch
from torch import nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride):
        super(ResBlk,self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        if ch_in!=ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )
        else:
            self.extra = nn.Sequential()

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        print(f"self.bn1(self.conv1(x)) is {out.shape}")
        out = self.bn2(self.conv2(out))
        print(f"self.bn2(self.conv2(out)) is {out.shape}")
        out = self.extra(x)+out
        print(f"self.extra(x) is {self.extra(x).shape}")
        print(f"self.extra(x)+out is {out.shape}")
        return out
def main():
    blk = ResBlk(64,128,stride = 2)
    tmp = torch.randn(2,64,32,32)
    out = blk(tmp)
    print(out.shape)
if __name__ == '__main__':
    main()
