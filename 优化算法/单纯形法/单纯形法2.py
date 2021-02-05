# https://blog.csdn.net/qq_41133375/article/details/105620784
import numpy as np
#定义线性回归系数模型
def pivot(d,bn):
    l = list(d[0][:-2])
    jnum = l.index(max(l)) #转入编号
    m = []
    for i in range(bn):
        if d[i][jnum] == 0:
            m.append(0.)
        else:
            m.append(d[i][-1]/d[i][jnum])
    inum = m.index(min([x for x in m[1:] if x!=0]))  #转出下标
    s[inum-1] = jnum
    r = d[inum][jnum]
    d[inum] /= r
    for i in [x for x in range(bn) if x !=inum]:
        r = d[i][jnum]
        d[i] -= r * d[inum]
#定义基变量函数
def solve(d,bn):
    flag = True
    while flag:
        if max(list(d[0][:-1])) <= 0: #直至所有系数小于等于0
            flag = False
        else:
            pivot(d,bn)
def printSol(d,cn):
    for i in range(cn - 1):
        if i in s:
            print("x"+str(i)+"=%.2f" % d[s.index(i)+1][-1])
        else:
            print("x"+str(i)+"=0.00")
    print("objective is %.2f"%(-d[0][-1]))
d = np.loadtxt("test.txt", dtype=np.float)
(bn,cn) = d.shape
s = list(range(cn-bn,cn-1)) #基变量列表
solve(d,bn)
printSol(d,cn)