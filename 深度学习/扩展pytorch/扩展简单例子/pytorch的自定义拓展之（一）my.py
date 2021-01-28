import torch
# 本例子所采用的数学公式是：
# y = (x-1)*(x-1)

class My(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x):  # input_x是一个tensor，不再是一个标量
        ctx.save_for_backward(input_x)
        print('正在前向传播')
        output = input_x
        print(f"output is {output}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print('正在反向传播')
        input_x, = ctx.saved_tensors
        # 获取前面保存的参数,也可以使用self.saved_variables  #input_x前面的逗号是不能丢的
        grad_x = input_x.clone()
        print(f"grad_x is {input_x}")
        return grad_x


def sqrt_and_inverse_func(input_x):
    return My.apply(input_x)  # 对象调用


x = torch.tensor([4.0], requires_grad=True)  # tensor
y = x*x
sqrt_and_inverse_func(y).backward()


print(x.grad)
'''运行结果为：
开始前向传播
开始反向传播
tensor([1.1547, 1.0607, 1.0328])
'''