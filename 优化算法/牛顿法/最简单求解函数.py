
# 求解 x**2 = m
# https://zhuanlan.zhihu.com/p/105265432
def newton(m):
    x0 = m/2 #初始点，也可以是别的值
    x1 = x0/2 + m/(x0*2)
    while abs(x1-x0)>1e-5:
        x0 = x1
        x1 = x0/2 + m/(x0*2)
    return x1
# 输出精确到小数点后四位
print( '%.4f'%newton(2) )