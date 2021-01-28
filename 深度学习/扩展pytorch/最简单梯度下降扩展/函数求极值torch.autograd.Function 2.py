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
      f,b = 0,0
      @staticmethod
      def forward(ctx, x):
            ctx.save_for_backward(x)
            y = f1(x)
            F1.f += 1
            print(f"第{F1.f}次向前\nx is {x}  ")
            print(f"y is {y}  ")
            return y

      @staticmethod
      def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            grad_x = df1(x)
            print(f"y开始反向传播 -------------------------\n")
            print(f"x is {x} \n y grad_output is {grad_output}  grad_x is {grad_x}\n")

            return grad_x
class F2(torch.autograd.Function):
      @staticmethod
      def forward(ctx, y):
            ctx.save_for_backward(y)
            z = f2(y)
            print(f"z is {z} ")
            return z

      @staticmethod
      def backward(ctx, grad_output):
            y, = ctx.saved_tensors
            grad_y = df2(y)
            print(f"z开始反向传播 ----------------------------\n")
            F1.b += 1
            print(f"第{F1.b}次向后")
            print(f"y is {y} z grad_output is {grad_output}  grad_y is {grad_y}\n")
            return grad_y


def plotf(loss):
      x = range(len(loss))
      plt.plot(x,loss)
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      plt.show()



def main():
      x = torch.tensor([1.],requires_grad=True)
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

      print(f"x is {x},函数最小值是：{f2(f1(x))} \n")
      plotf(losses)



if __name__ == '__main__':
    main()

