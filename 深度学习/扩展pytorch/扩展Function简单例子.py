import torch
from torch.autograd import Function


# 类需要继承Function类，此处forward和backward都是静态方法
class MultiplyAdd(Function):

    @staticmethod
    def forward(ctx, x,w,  b):
        ctx.save_for_backward(w, x)  # 保存参数,这跟前一篇的self.save_for_backward()是一样的
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):  # 获取保存的参数,这跟前一篇的self.saved_variables()是一样的
        w, x = ctx.saved_tensors
        print("grad_output:", grad_output)
        print("=======================================")
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_x, grad_w, grad_b  # backward输入参数和forward输出参数必须一一对应


x = torch.ones(1, requires_grad=True)  # x 是1，所以grad_w=1
w = torch.rand(1, requires_grad=True)  # w 是随机的，所以grad_x=随机的一个数
b = torch.rand(1, requires_grad=True)  # grad_b 恒等于1

print('开始前向传播')
z = MultiplyAdd.apply(x,w,  b)  # forward,这里的前向传播是不一样的，这里没有使用函数去包装自定义的类，而是直接使用apply方法
print('开始反向传播')
z.backward()  # backward

print(x.grad, w.grad, b.grad)