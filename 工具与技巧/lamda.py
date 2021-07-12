# 普通python函数 lambda 定义了一个匿名函数
def func(a, b, c):
    return a + b + c

print(func(1, 2, 3))
# 返回值为6

# lambda匿名函数
f = lambda a, b, c: a + b + c

print(f(1, 2, 3))

# 返回结果为6

foo = [1, 2, 3, 4, 5, 6, 7, 8, 9]
f = list(map(lambda x: x * x, foo))
print (f)


