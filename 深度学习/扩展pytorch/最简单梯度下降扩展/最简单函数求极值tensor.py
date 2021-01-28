import torch
import torch.optim
import matplotlib.pyplot as plt


def f(x):
      return x**2-3

def df(x):
      return 2*x
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
            y = f(x)
            optimizer.zero_grad()
            y.backward()
            optimizer.step()
            losses.append(f(x))
            print(losses[i])
      y = f(x)
      print("函数最小值是： ",y)
      plotf(losses)



if __name__ == '__main__':
    main()

