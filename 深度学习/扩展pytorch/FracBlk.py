import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import gamma #分数阶微分用

diff_order=1.9

class FracF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        # x:[200,784],  w:[200,784], b[200,1]
        ctx.save_for_backward(x, w, b)
        # print(f"开始正向传播")
        x = x @ w.t() + b
        # X torch.Size([210, 200])
        return x, w, b

    @staticmethod
    def backward(ctx, grad_x, grad_w, grad_b):
        O, w, b = ctx.saved_tensors
        # O [200,784]  w[200,784]
        # error = np.dot(params['W3'].T, error) * self.sigmoid(params['X2'], derivative=True)
        # change_w['W2'] = np.outer(error, params['O1'])
        grad_x_new = grad_x @ w

        # grad_x_new [210,784]
        grad_w_new = grad_x.t() @ O
        # grad_w_new [200,784]
        #######################   根据论文 （18）式 ####################################
        grad_w_new = grad_w_new * (abs(w) ** (1 - diff_order)) / gamma(2 - diff_order)
        # grad_b_new = grad_b * (abs(b)**(1-diff_order))/gamma(2-diff_order)
        #################################################################################

        # print(f"开始反向传播 grad_x is {grad_x.shape}")
        return grad_x_new, grad_w_new, grad_b


class FracBlk(nn.Module):
    def forward(self, x, w1, b1):
        X1, w11, b11 = FracF.apply(x, w1, b1)
        return X1

class FracBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride):
        super(FracBlk,self).__init__()
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
    blk = FracBlk(64,128,stride = 2)
    tmp = torch.randn(2,64,32,32)
    out = blk(tmp)
    print(out.shape)
if __name__ == '__main__':
    main()
