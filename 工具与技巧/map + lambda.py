def square(x) :         # 计算平方数
    return x ** 2
map(square, [1,2,3,4,5])    # 计算列表各个元素的平方
# <map object at 0x100d3d550>     # 返回迭代器
list(map(square, [1,2,3,4,5]))   # 使用 list() 转换为列表
# [1, 4, 9, 16, 25]
list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))   # 使用 lambda 匿名函数
# [1, 4, 9, 16, 25]


from scipy.special import gamma #分数阶微分用
der_order,nabla_w_ ,weights= 1.5,[2,2],[3,3]
a1 = lambda x,y: x * y**(1-der_order)/gamma(2-der_order)
print(f"a1 = ",a1)
a2 = list(map(lambda x,y: x * y**(1-der_order)/gamma(2-der_order), nabla_w_, weights))
print(f"a2 = ",a2)