import torch

# 图片 1-4 的识别

logits = [[0.1,0.1,0.2,0.9],
          [0.2,0.8,0.1,0.1],
          [0.7,0.1,0.1,0.1]]

logits = torch.tensor(logits)

pred1 = logits.argmax(dim=0)
pred2 = logits.argmax(dim=1)
print(f"pred1 is {pred1}")
print(f"pred2 is {pred2}")
# pred1 is tensor([2, 1, 0, 0])
# pred2 is tensor([3, 1, 0])
lable = torch.tensor([2,1,0])
correct = torch.eq(pred2,lable)
print(f"correct is {correct} {correct.sum().item()}")
# correct is tensor([ True,  True, False, False]) 2.0