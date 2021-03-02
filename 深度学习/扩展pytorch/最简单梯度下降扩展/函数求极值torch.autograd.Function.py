import torch
import torch.optim

import matplotlib.pyplot as plt

def f(x):
      return x**2-3

class MyF(torch.autograd.Function):
      @staticmethod
      def forward(ctx, x):
            ctx.save_for_backward(x)
            return f(x)

      @staticmethod
      def backward(ctx, *grad_output):
            x, = ctx.saved_tensors

            grad_x = 2*x
            print(f"***x is {x} \ngrad_output is {grad_output} \n grad_x is {grad_x}")
            print(f"开始反向传播 grad_x is {grad_x}")
            return grad_x


def plotf(loss):
      x = range(len(loss))
      plt.plot(x,loss)
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      plt.show()



def main():
      x = torch.tensor([15.],requires_grad=True)
      optimizer = torch.optim.SGD([x,],lr = 0.1,momentum=0.9)
      steps = 400
      losses = []

      for i in range(steps):
            y = MyF.apply(x)
            optimizer.zero_grad()
            y.backward()
            optimizer.step()
            losses.append(y)
            # print(losses[i])
      y = f(x)
      print("函数最小值是： ",y)
      plotf(losses)

if __name__ == '__main__':
    main()

