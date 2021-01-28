
import numpy as np

a= [[1],[2],[3]]
b = [[4],[5]]
c = np.outer(b, a)
print(c)

# 1-D array
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result_ab = np.dot(a, b)
result_ba = np.dot(b, a)
print('result_ab: %s' %(result_ab))
print('result_ba: %s' %(result_ba))