import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable

class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        a = torch.ones_like(input)
        b = -torch.ones_like(input)
        output = torch.where(input>=0,a,b)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        input, = ctx.saved_tensors
        input_abs = torch.abs(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = torch.where(input_abs<=1,ones, zeros)
        return input_grad

class BinarizedModule(nn.Module):
    def __init__(self):
        super(BinarizedModule, self).__init__()
        self.BF = BinarizedF()
    def forward(self,input):
        print(input.shape)
        output =self.BF.apply(input)
        return output

a = Variable(torch.randn(4,480,640), requires_grad=True)
output = BinarizedModule().forward(a)
output.backward(torch.ones(a.size()))
print(a)
print(a.grad)