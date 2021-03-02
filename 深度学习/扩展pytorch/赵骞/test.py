import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
w1, b1 = torch.randn(120, 16*5*5).to(device), torch.zeros(120, requires_grad=True).to(device)


w1.requires_grad = True
print(w1.is_leaf)