# https://www.jiqizhixin.com/articles/2018-10-17-20

# https://www.jiqizhixin.com/articles/2019-02-12-10

# https://www.jiqizhixin.com/articles/2019-01-23-15
# !/usr/bin/env python
from numpy import *


class Model(object):

    def __init__(self, X, y, C, toler, kernel_param):
        self.X = X
        self.y = y
        self.C = C
        self.toler = toler
        self.kernel_param = kernel_param
        self.m = shape(X)[0]
        self.mapped_data = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.mapped_data[:, i] = gaussian_kernel(self.X, X[i, :], self.kernel_param)
        self.E = mat(zeros((self.m, 2)))
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0


def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as fd:
        for line in fd.readlines():
            nums = line.strip().split(',')
            X_temp = []
            for i in range(len(nums)):
                if i == len(nums) - 1:
                    y.append(float(nums[i]))
                else:
                    X_temp.append(float(nums[i]))
            X.append(X_temp)
    return mat(X), mat(y)

def gaussian_kernel(X, l, kernel_param):
    sigma = kernel_param
    m = shape(X)[0]
    mapped_data = mat(zeros((m, 1)))
    for i in range(m):
        mapped_data[i] = exp(-sum((X[i, :] - l).T * (X[i, :] - l) / (2 * sigma ** 2)))
    return mapped_data

def clip_alpha(L, H, alpha):
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    return alpha

def calc_b(b1, b2):
    return (b1 + b2) / 2

def calc_E(i, model):
    yi = float(model.y[i])
    gxi = float(multiply(model.alphas, model.y).T * model.mapped_data[:, i] + model.b)
    Ei = gxi - yi
    return Ei

def select_j(Ei, i, model):
    nonzero_indices = nonzero(model.E[:, 0].A)[0]
    Ej = 0
    j = 0
    max_delta = 0
    if len(nonzero_indices) > 1:
        for index in nonzero_indices:
            if index == i:
                continue
            E_temp = calc_E(index, model)
            delta = abs(E_temp - Ei)
            if delta > max_delta:
                max_delta = delta
                Ej = E_temp
                j = index
    else:
        j = i
        while j == i:
            j = int(random.uniform(0, model.m))
        Ej = calc_E(j, model)
    return j, Ej

def iterate(i, model):
    yi = model.y[i]
    Ei = calc_E(i, model)
    model.E[i] = [1, Ei]
    # 如果alpahi不满足KKT条件, 则进行之后的操作, 选择alphaj, 更新alphai与alphaj, 还有b
    if (yi * Ei > model.toler and model.alphas[i] > 0) or (yi * Ei < -model.toler and model.alphas[i] < model.C):
        # alphai不满足KKT条件
        # 选择alphaj
        j, Ej = select_j(Ei, i, model)
        yj = model.y[j]
        alpha1old = model.alphas[i].copy()
        alpha2old = model.alphas[j].copy()
        eta = model.mapped_data[i, i] + model.mapped_data[j, j] - 2 * model.mapped_data[i, j]
        if eta <= 0:
            return 0
        alpha2new_unclip = alpha2old + yj * (Ei - Ej) / eta
        if yi == yj:
            L = max(0, alpha2old + alpha1old - model.C)
            H = min(model.C, alpha1old + alpha2old)
        else:
            L = max(0, alpha2old - alpha1old)
            H = min(model.C, model.C - alpha1old + alpha2old)
        if L == H:
            return 0
        alpha2new = clip_alpha(L, H, alpha2new_unclip)
        if abs(alpha2new - alpha2old) < 0.00001:
            return 0
        alpha1new = alpha1old + yi * yj * (alpha2old - alpha2new)
        b1new = -Ei - yi * model.mapped_data[i, i] * (alpha1new - alpha1old) \
                - yj * model.mapped_data[j, i] * (alpha2new - alpha2old) + model.b
        b2new = -Ej - yi * model.mapped_data[i, j] * (alpha1new - alpha1old) \
                - yj * model.mapped_data[j, j] * (alpha2new - alpha2old) + model.b
        model.b = calc_b(b1new, b2new)
        model.alphas[i] = alpha1new
        model.alphas[j] = alpha2new
        model.E[i] = [1, calc_E(i, model)]
        model.E[j] = [1, calc_E(j, model)]
        return 1
    return 0

def smo(X, y, C, toler, iter_num, kernel_param):
    model = Model(X, y.T, C, toler, kernel_param)
    changed_alphas = 0
    current_iter = 0
    for i in range(model.m):
        changed_alphas += iterate(i, model)
        print("iter:%d i:%d,pairs changed %d" %(current_iter, i, changed_alphas))
    current_iter += 1
    print('start...')
    while current_iter < iter_num and changed_alphas > 0:
        changed_alphas = 0
        # 处理支持向量
        alphas_indice = nonzero((model.alphas.A > 0) * (model.alphas.A < C))[0]
        for i in alphas_indice:
            changed_alphas += iterate(i, model)
            print("iter:%d i:%d,pairs changed %d"
                  %(current_iter, i, changed_alphas))
        current_iter += 1
    return model.alphas, model.b