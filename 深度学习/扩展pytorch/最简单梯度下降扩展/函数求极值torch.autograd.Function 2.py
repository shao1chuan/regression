import torch
import torch.optim

import matplotlib.pyplot as plt

def f1(x):
      return x**2+2
def df1(x):
      return 2*x

def f2(y):
      return y**2+3
def df2(y):
      return 2*y


class F1(torch.autograd.Function):
      @staticmethod
      def forward(ctx, x):
            ctx.save_for_backward(x)
            return f1(x)

      @staticmethod
      def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            grad_x = df1(x)
            print(f"y开始反向传播 grad_x is {grad_x}")
            return grad_x
class F2(torch.autograd.Function):
      @staticmethod
      def forward(ctx, y):
            ctx.save_for_backward(y)
            z = f2(y)
            print(f"z is {z}")
            return z

      @staticmethod
      def backward(ctx, grad_output):
            y, = ctx.saved_tensors
            grad_y = df2(y)
            print(f"z开始反向传播 grad_y is {grad_y}")
            return grad_y


def plotf(loss):
      x = range(len(loss))
      plt.plot(x,loss)
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      plt.show()



def main():
      x = torch.tensor([2.],requires_grad=True)
      optimizer = torch.optim.SGD([x,],lr = 0.1,momentum=0.9)
      steps = 400
      losses = []

      for i in range(steps):
            y = F1.apply(x)
            z = F2.apply(y)
            optimizer.zero_grad()
            z.backward()
            optimizer.step()
            losses.append(z)
            # print(losses[i])

      print(f"x is {x},函数最小值是：{f2(f1(x))} ")
      plotf(losses)



if __name__ == '__main__':
    main()

