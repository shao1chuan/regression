import torch

x = torch.linspace(1,5,10)
y = x.view(-1,1)
print(x)

for xi,yi in zip(x,y):
    # i = i.unsqueeze(1)
    # i=i.view(-1,1)
    print(xi.shape,yi.shape)