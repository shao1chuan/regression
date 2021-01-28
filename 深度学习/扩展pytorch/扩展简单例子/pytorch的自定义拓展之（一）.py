# class My_Function(Function):
#     def forward(self, inputs, parameters):
#         self.saved_for_backward = [inputs, parameters]
#         # output = [对输入和参数进行的操作，其实就是前向运算的函数表达式]
#         return output
#
#     def backward(self, grad_output):
#         inputs, parameters = self.saved_tensors  # 或者是self.saved_variables
#         # grad_inputs = [求函数forward(input)关于 parameters 的导数，
#         其实就是反向运算的导数表达式] * grad_output
#         return grad_input

import torch
# 本例子所采用的数学公式是：
# y = (x-1)*(x-1)

class sqrt_and_inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x):  # input_x是一个tensor，不再是一个标量
        ctx.save_for_backward(input_x)
        print('正在前向传播')
        output = torch.pow(input_x, 2)   # 函数z
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print('正在反向传播')
        input_x, = ctx.saved_tensors
        # 获取前面保存的参数,也可以使用self.saved_variables  #input_x前面的逗号是不能丢的
        grad_x = grad_output*2*(input_x-1)
        return grad_x


def sqrt_and_inverse_func(input_x):
    return sqrt_and_inverse.apply(input_x)  # 对象调用


x0 = torch.tensor([1.0], requires_grad=True)  # tensor
print('开始前向传播')
y = sqrt_and_inverse_func(x0)
print('开始反向传播')
y.backward()
print(x0.grad)
'''运行结果为：
开始前向传播
开始反向传播
tensor([1.1547, 1.0607, 1.0328])
'''