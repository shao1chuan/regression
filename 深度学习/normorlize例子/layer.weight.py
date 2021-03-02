import torch

x= torch.rand(1,5,28,28)

layer = torch.nn.BatchNorm2d(5)
out = layer(x)
print(f"layer.weight is {layer.weight}")
print(f"layer.bias is {layer.bias}")

for i in range(100):
    out = layer(x)
print(f"100 layer.weight is {layer.weight}")
print(f"100 layer.bias is {layer.bias}")
