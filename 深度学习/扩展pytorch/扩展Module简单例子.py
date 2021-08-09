import torch
from torch.autograd import Function
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import math
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple
import torch.optim as optim
from scipy.special import gamma

# 类需要继承Function类，此处forward和backward都是静态方法
class Frac(Function):

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, diff_order = None) -> Tensor:
       ctx.save_for_backward(input, weight, diff_order)  # 保存参数,这跟前一篇的self.save_for_backward()是一样的
       # output = w * x + b
       print("开始正向传播")
       output = torch._C._nn.linear(input, weight, bias)
       return output

    @staticmethod
    def backward(ctx, grad_output):  # 获取保存的参数,这跟前一篇的self.saved_variables()是一样的
        x, w, diff_order = ctx.saved_tensors
        print("diff_order is ---------------------", diff_order)
        print("开始反向传播")
        grad_w = grad_output.t().mm(x)
        grad_x = grad_output.mm(w)
        grad_b = grad_output.sum(0).squeeze(0)
        if diff_order:
            grad_w = grad_w * (abs(w) ** (1 - diff_order)) / gamma(2 - diff_order)
        return grad_x, grad_w, grad_b, None  # backward输入参数和forward输出参数必须一一对应

class FracLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, diff_order: float = 1.0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FracLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if diff_order != 1.0:
            self.diff_order = Parameter(torch.tensor(diff_order))
        else:
            self.register_parameter('diff_order', None)
        self.reset_parameters()

        __constants__ = ['in_features', 'out_features']
        in_features: int
        out_features: int
        weight: Tensor
        diff_order: float

    def reset_parameters(self) -> None:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)


    def extra_repr(self) -> str:
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )


    def forward(self, input: Tensor) -> Tensor:
        out = Frac.apply(input, self.weight, self.bias, self.diff_order)
        return out



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = FracLinear(28 * 28, 256, diff_order=0.91)
        self.fc2 = FracLinear(256, 64)
        self.fc3 = FracLinear(64, 10, diff_order=1.1)

    def forward(self, x):
        # x: [b, 4, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x


def main():
    net = Net()
    tmp = torch.randn(2, 28*28)
    out = net(tmp)
    # print(out.shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    labels = torch.tensor([1, 9])
    for i  in range(10):

        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(tmp)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()


