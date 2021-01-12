import numpy as np
import random

def lossF(points, w, b):
    loss = 0
    for i in range(0,len(points)):
        loss += (w * points[i][0] + b - points[i][1]) ** 2
    return loss / len(points)


def step(w, b, points, rate):
    wg = random.random()
    bg = random.random()
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        wg += 2 * x * (x * w + b - y)
        bg += 2 * (x * w + b - y)
    w1 = w - wg * rate / N
    b1 = b - bg * rate / N
    return [w1, b1]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")

    w = 0
    b = 0
    num_iterations = 1000
    rate = 0.0001
    for i in range(0, num_iterations):
        w, b = step(w, b, points, rate)
        print(w, b)
    loss = lossF(points,w,b)
    print("the loss is :"+str(loss))
    print(points.shape)


if __name__ == '__main__':
    run()
