import torch
import numpy as np

x= torch.randn(100,16)+0.5
# u = 0.5,sigma =1

layer = torch.nn.BatchNorm1d(16)
out = layer(x)
print(f"layer.running_mean is {layer.running_mean}")
print(f"layer.running_var is {layer.running_var}")

for i in range(100):
    out = layer(x)
print(f"100 layer.running_mean is {layer.running_mean}")
print(f"100 layer.running_var is {layer.running_var}")
