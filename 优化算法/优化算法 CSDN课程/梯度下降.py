import numpy as np
import matplotlib.pyplot as plt

#生成2行100列数据
X = np.array([np.ones(100), np.random.rand(100)])

y = np.dot([4, 3], X) + np.random.rand(100)
print(X.T.shape,X.shape,y.shape)
plt.scatter(X[1, :], y)
plt.show()
alpha = 0.01
num_iters = 1000


def gradient_descent(theta, X, y, alpha, num_iters):
    loss_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters, 2))

    m = len(y)
    loss = 0
    for i in range(num_iters):
        y_pred = np.dot(theta, X)

        theta = theta - alpha/m*np.dot(y_pred - y, X.T)
        loss = 1/(2*m)* np.sum(np.square(y_pred - y))

        if i % 100 == 0:
            print("Iterating:{}".format(i))
            print("theta:{}".format(theta))
            print("loss:{}\n".format(loss))

        theta_history[i, :] = theta
        loss_history[i] = loss
    return theta, theta_history, loss_history


theta_init = np.random.randn(2)
theta, theta_history, loss_history = gradient_descent(theta_init, X, y, alpha, num_iters)


