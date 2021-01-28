import torch
import matplotlib.pyplot as plt
import numpy as np

class MyReLU(torch.autograd.Function):
    """
    我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，
    并完成张量的正向和反向传播。
    """
    @staticmethod
    def forward(ctx, x):
        """
        在正向传播中，我们接收到一个上下文对象和一个包含输入的张量；
        我们必须返回一个包含输出的张量，
        并且我们可以使用上下文对象来缓存对象，以便在反向传播中使用。
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们接收到上下文对象和一个张量，
        其包含了相对于正向传播过程中产生的输出的损失的梯度。
        我们可以从上下文对象中检索缓存的数据，
        并且必须计算并返回与正向传播的输入相关的损失的梯度。
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        print(grad_x.shape)
        grad_x[x < 0] = 0
        return grad_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# N是批大小；D_in 是输入维度；
# H 是隐藏层维度；D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10
# 产生输入和输出的随机张量
torch.manual_seed(1)
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# 产生随机权重的张量
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)
losses = []
learning_rate = 1e-6
steps = 50
for t in range(steps):
    # 正向传播：使用张量上的操作来计算输出值y；
    # 我们通过调用 MyReLU.apply 函数来使用自定义的ReLU
    x1 = x @ w1
    x1 = MyReLU.apply(x1)
    x2 = x1@ w2
    x2 = MyReLU.apply(x2)
    # y_pred = MyReLU.apply(x2)
    # 计算并输出loss
    y_pred = x2
    loss = (y_pred - y).pow(2).sum()
    losses.append(loss)
    print(t, loss.item())
    # 使用autograd计算反向传播过程。
    loss.backward()
    with torch.no_grad():
    # 用梯度下降更新权重
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 在反向传播之后手动清零梯度
        w1.grad.zero_()
        w2.grad.zero_()
x = range(len(losses))
plt.plot(x, losses, label="losses")
# plt.legend(loc='best bbbbbbbbbb')
plt.show()


# 499 480.34185791015625