import torch
# 本例子所采用的数学公式是：
# z=sum(sqrt(x*x-1))
# 这个时候x是一个向量，x=[x1,x2,x3]
# z'(x)=x/sqrt(x*x-1)
class sqrt_and_inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x):  # input_x是一个tensor，不再是一个标量
        ctx.save_for_backward(input_x)
        output = torch.sum(torch.sqrt(torch.pow(input_x, 2) - 1))  # 函数z
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print('正在反向传播')
        input_x, = ctx.saved_tensors
        # 获取前面保存的参数,也可以使用self.saved_variables  #input_x前面的逗号是不能丢的
        grad_x = grad_output * (torch.div(input_x, torch.sqrt(torch.pow(input_x, 2) - 1)))
        return grad_x


def sqrt_and_inverse_func(input_x):
    return sqrt_and_inverse.apply(input_x)  # 对象调用


x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)  # tensor
print('开始前向传播')
z = sqrt_and_inverse_func(x)
print(z)
print('开始反向传播')
z.backward()
print(x.grad)
'''运行结果为：
开始前向传播
开始反向传播
tensor([1.1547, 1.0607, 1.0328])
'''