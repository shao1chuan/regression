import numpy as np

a = np.array([[1,2,3]])
D = np.diag([2,3,4])
print(a.dot(D))

a = np.array([[2,0,0,0],
              [0,4,0,0],
              [0,0,3,0],
              [0,0,0,1]])
print(np.linalg.inv(a))
