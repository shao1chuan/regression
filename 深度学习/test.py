import torch

A = torch.empty(4, 1, 2, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A.to(device)
print(A.device)
A = A.to(device)
print(A.device)
