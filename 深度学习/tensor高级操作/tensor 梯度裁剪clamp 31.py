import numpy as np
import torch
torch.manual_seed(1)
a1 = torch.rand(2,3)
# tensor([[0.7576, 0.2793, 0.4031],
#         [0.7347, 0.0293, 0.7999]])
print(f"{a1.clamp(0.4)}")
# tensor([[0.7576, 0.4000, 0.4031],
#         [0.7347, 0.4000, 0.7999]])

print(f"{a1.clamp(0.4,0.7)}")
# tensor([[0.7000, 0.4000, 0.4031],
#         [0.7000, 0.4000, 0.7000]])



