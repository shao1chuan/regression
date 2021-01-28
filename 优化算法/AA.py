#coding:utf-8
# https://mp.weixin.qq.com/s/ac-CgZj-avmPBraVvQUuBQ
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
class Function():
    def __init__(self):
        self.points_x = np.linspace(-20, 20, 1000)
        self.points_y = self.f(self.points_x)

    def f(self,x):
        return (0.15*x)**2 + np.cos(x) + np.sin(3*x)/3 + np.cos(5*x)/5 + np.sin(7*x)/7

    def df(self,x):
        return (9/200)*x - np.sin(x) -np.sin(5*x) + np.cos(3*x) + np.cos(7*x)


    def ddf(self,x):
        return (9/200) - np.cos(x) -3*np.sin(x) - 5*np.cos(5*x) -7* np.sin(7*x)