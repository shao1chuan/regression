import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride):
        super(ResBlk,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

        )

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self,x):
        """

        :param x: [b,ch,h,w]
        :return:
        """
        # x :2,3,32,32
        out =self.net(x)
        print(f"ResBlk after net :{out.shape}")
        # ([2, 16, 8, 8])
        out = self.extra(x)+out
        print(f"ResBlk after extra :{out.shape}")
        return out
def main():
    testdata  = torch.randn(2,3,32,32)
    test = ResBlk(3,16,stride=2)
    print(test(testdata).shape)
if __name__ == '__main__':
    main()
