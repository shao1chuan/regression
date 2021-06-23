import torch
import  torch.nn.functional as F

x = torch.randn(1,10)
w = torch.randn(1,10,requires_grad=True)

o = torch.sigmoid(x@w.t())

loss = F.mse_loss(torch.ones(1,1),o)
loss.backward()
print(w)
