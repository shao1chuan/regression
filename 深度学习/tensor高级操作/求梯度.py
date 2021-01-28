import torch

# x = torch.randn((1),requires_grad=True)
# print(x.grad)#此时还为None
# y = x + 2
# y.backward()  #反向传播计算梯度
# print(x.grad)     #此时，grad属性为tensor([1.])，梯度为１表明该函数沿着x轴正方向会增长的最快
# y.backward()  #再来一次反向传播计算梯度
# print(x.grad)      #此时，grad属性为tensor([２.])，表明梯度累加了，所以每次计算之前应当将梯度清零
#
#
# import torch as t
# x = t.randn(2, 1, requires_grad=True)
# y = x + 2
# y.backward()
# #此时执行程序会报错：RuntimeError: grad can be implicitly created only for scalar outputs
# #因为y此时不是一个标量，所以应当加入gradient参数：y.backward(t.FloatTensor(y.size())．fill_(1))


x = torch.randn((1),requires_grad=True)  #x是叶子张量
print(f"x.is_leaf {x.is_leaf}"  )                          #True
y = x**2 #y是通过＂＋＂操作计算来的，所以不是叶子张量
print(f"y.is_leaf {y.is_leaf } "  )
                            #False
y.backward()   #反向传播计算梯度
print(f"x.grad {x.grad } "  ) #此时，grad属性为tensor([1.])，梯度为１表明该函数沿着x轴正方向会增长的最快
print(f"y.grad {y.grad } "  )  #None

# #使用retain_grad()函数
# y.retain_grad()
# y.backward()   #再反向传播计算梯度
# print(f"x.grad {x.grad } "  ) #累加为tensor([2.])
# print(f"y.grad {y.grad } "  ) #为tensor([1.])