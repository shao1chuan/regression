import visdom
import numpy as np
import time

vis = visdom.Visdom()

x, y = 0, 0
X = np.arange(1,10,1)
win = vis.line(np.array([y]), np.array([x]))
for i in range(10):
    x = np.array([i])

    time.sleep(0.1)
    y = np.random.randn(1)
    vis.line(y, x, win, update="append")

vis.line(2*X, X, win, update="new")
