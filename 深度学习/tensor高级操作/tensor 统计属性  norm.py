import numpy as np
import torch

a1 = torch.full([8],1)
a2 = a1.reshape(2,4)
a3 = a1.reshape(2,2,2)
print("a1.norm(1),a2.norm(1),a3.norm(1)",a1.norm(1),a2.norm(1),a3.norm(1))
print("a1.norm(2),a2.norm(2),a3.norm(2)",a1.norm(2),a2.norm(2),a3.norm(2))

a2.norm(1,dim=1)
print("a2.norm(1,dim=1) ",a2.norm(1,dim=1))
print("a2.norm(2,dim=2) ",a2.norm(2,dim=0))
print("a3.norm(1,dim=1) ",a3.norm(1,dim=1))

