

import torch
input = [
    [2, 3, 4, 5, 0, 0],
    [1, 4, 3, 0, 0, 0],
    [4, 2, 2, 5, 7, 0],
    [1, 0, 0, 0, 0, 0]
]
input = torch.tensor(input)
#注意index的类型
length = torch.LongTensor([[4],[3],[5],[1]])
#index之所以减1,是因为序列维度是从0开始计算的
out = torch.gather(input, 1, length-1)
print(out)