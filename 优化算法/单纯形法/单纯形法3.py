# https://blog.csdn.net/qq_41133375/article/details/105620784

#导入包
from scipy import optimize
import numpy as np
#确定c,A_ub,B_ub
c = np.array([3,4])
A_ub = np.array([[2,1],[1,3]])
B_ub = np.array([40,30])
#求解
res =optimize.linprog(-c,A_ub,B_ub)
print(res)