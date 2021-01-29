import numpy as np
import torch
# 只取后两位运算
a1 = torch.rand(4,3,28,64)
a2 = torch.rand(4,3,64,32)
print(f"a1@a2 = {(a1@a2).shape}")
# 只取后两位运算  broadcast
a1 = torch.rand(4,3,28,64)
a2 = torch.rand(4,1,64,32)
print(f"a1@a2 = {(a1@a2).shape}")

# 平方  开方  开方导数
a1 = torch.full([3,4],4)
print(f"a1 = {a1} \n a1.pow(2) is {a1.pow(2)}")
print(f"a1.sqrt() = {a1.sqrt()} \n a1.rsqrt() is {a1.rsqrt()}\n a1**0.5 is {a1**0.5} ")

# 指数对数
print(f"a1.exp() = {a1.exp()}\n a1.log() is {a1.log2()}")

a3 = torch.tensor(3.14)
print(f"a3.ceil() {a3.ceil()} a3.floor() {a3.floor()}a3.trunc() {a3.trunc()} a3.frac() {a3.frac()}")
# a3.ceil() 4.0 a3.floor() 3.0a3.trunc() 3.0 a3.frac() 0.1400001049041748
a4 = torch.tensor(3.54)
print(f"a3.round() {a3.round()} a4.round() {a4.round()}")



