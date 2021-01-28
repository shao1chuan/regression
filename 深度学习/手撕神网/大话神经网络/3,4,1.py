# https://www.bilibili.com/video/BV1XE411C7mS?from=search&seid=12118666254786225732

import numpy as np

# X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# y = np.array([[0],[1],[1],[0]])
# y = np.array([0,1,1,0])
# y = y.reshape(4,1)
# [4,1]
# x_test = np.array([[0,0,1],[0,1,1],[1,0,1],[1,0,0]])



Lr = 0.01
size = np.array([3,4,1])
y = [0]

W = []
w1 = np.random.random((3,4))
w2 = np.random.random((4,1))
W.append(0)
W.append(w1)
W.append(w2)

B = []
b1 = np.random.random((1,4))
b2 = np.random.random((1,1))
B.append(0)
B.append(b1)
B.append(b2)

O = []
O0 =  np.array([0,0,1])
# [3,1]
O1 = np.zeros_like(b1)
O2 = np.zeros_like(b2)
O.append(O0)
O.append(O1)
O.append(O2)

O_grade = []
Og1 = np.zeros_like(b1)
Og2 = np.zeros_like(b2)
O_grade.append(0)
O_grade.append(Og1)
O_grade.append(Og2)



W_grade = []
w1_grade = np.zeros_like(w1)
w2_grade = np.zeros_like(w2)
W_grade.append(0)
W_grade.append(w1_grade)
W_grade.append(w2_grade)
# print("W_grade: ",W_grade)

B_grade = []
b1_grade = np.zeros_like(b1)
b2_grade = np.zeros_like(b2)
B_grade.append(0)
B_grade.append(b1_grade)
B_grade.append(b2_grade)



def sigmoid(x):
    return 1/(1 + np.exp(-x))
def desigmoid(x):
    return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)
def desoftmax(self, x, derivative=False):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

def forward(X):

    x1 = O[0]@w1+b1
    # [1,3]@[3,4]+[1,4]
    O[1] = sigmoid(x1)
    x2 = O[1]@w2+b2
    # [1,4]@[4,1]+[1,1]
    output = O[2] = sigmoid(x2)
    # output = O[2] = softmax(x2)
    return output


def lossF(output,y):
    error = 0.5*np.sum((output-y)**2)
    return error

def backward(output,y):

    O_grade[2] = (output - y)*desigmoid(O[2])
    # [1,1]
    W_grade[2] = O_grade[2]*O[1]
    # [4,1]
    B_grade[2] = O_grade[2]
    # print(f"O_grade[2] is{O_grade[2]} .W_grade[2] is {W_grade[2]}. B_grade[2] is {B_grade[2]} \n")
    O_grade[1] = (W[2]*O_grade[2]).T*desigmoid(O[1])
    # [4,1]@[1,1].T*[1,4]
    # [1,4]
    # W_grade[1] = O_grade[1].T*O[0]
    # print(f" O[0] is {O[0].T} \n O_grade[1] is {O_grade[1]} ")
    W_grade[1] = O[0].reshape(3,1)@O_grade[1]
    # print(f" O_grade[1] is {O_grade[1]} \n W_grade[1] is {W_grade[1]} ")
    B_grade[1] = O_grade[1]


def zeros():

    O = []
    O1 = np.zeros_like(size[1])
    O2 = np.zeros_like(size[2])
    O.append(O0)
    O.append(O1)
    O.append(O2)

    O_grade = []
    Og1 = np.zeros_like(size[1])
    Og2 = np.zeros_like(size[2])
    O_grade.append(0)
    O_grade.append(Og1)
    O_grade.append(Og2)

    W = []
    w1 = np.random.random((3, 4))
    w2 = np.random.random((4, 1))
    W.append(0)
    W.append(w1)
    W.append(w2)

    B = []
    b1 = np.random.random((4, 1))
    b2 = np.random.random((1, 1))
    B.append(0)
    B.append(b1)
    B.append(b2)

    W_grade = []
    w1_grade = np.zeros_like(w1)
    w2_grade = np.zeros_like(w2)
    W_grade.append(0)
    W_grade.append(w1_grade)
    W_grade.append(w2_grade)
    print("W_grade: ",W_grade)


    B_grade = []
    b1_grade = np.zeros_like(b1)
    b2_grade = np.zeros_like(b2)
    B_grade.append(0)
    B_grade.append(b1_grade)
    B_grade.append(b2_grade)

def step():
    for j,i in zip(W_grade,W):
        print(f" W_grade is {j} W is {i} \n")
        i = i-j


output = forward(O[0])
error = lossF(O[2],y)
zeros()
backward(output,y)
step()

