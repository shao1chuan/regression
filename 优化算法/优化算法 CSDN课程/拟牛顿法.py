import numpy as np
import matplotlib.pyplot as plt

X = np.array([np.ones(100), np.random.rand(100)])
y = np.dot([4, 3], X) + np.random.rand(100)
plt.scatter(X[1, :], y)
num_iters = 5
m = len(y)


def grad(theta):
    grad = 1 / m * (np.dot(theta, X) - y).dot(X.T)
    return grad


def mse_loss(theta):
    mse = 1 / (2 * m) * np.sum(np.square(np.dot(theta, X) - y))
    return mse


def quasi_newton(theta, X, y, nim_iters):
    n = X.shape[0]

    theta_history = np.zeros((num_iters + 1, n))
    loss_history = np.zeros(num_iters + 1)
    D_history = np.zeros((num_iters + 1, n, n))

    theta_history[0, :] = theta
    loss_history[0] = mse_loss(theta)
    D_history[0] = np.eye(n)

    for i in range(1, nim_iters + 1):
        g = grad(theta)
        D = D_history[i - 1]

        theta_prev = theta_history[i - 1]

        theta = theta_prev - g.dot(D)
        loss = mse_loss(theta)

        s = theta - theta_prev
        y_ = grad(theta) - grad(theta_prev)
        delD = np.dot(s.T, s) / np.dot(s, s.T) - (np.dot(D.T, y_.T).dot(np.dot(y_, D))) / (y_.dot(D).dot(y_.T))
        D_next = D + delD

        theta_history[i, :] = theta
        loss_history[i] = loss
        D_history[i] = D_next

        print("Iterating:{}".format(i))
        print("theta:{}".format(theta))
        print("loss:{}\n".format(loss))

    return theta, theta_history, loss_history, D_history


theta_init = np.random.random((1, 2))
theta, theta_history, loss_history, D_history = quasi_newton(theta_init, X, y, num_iters)
plt.plot(loss_history)

plt.scatter(X[1, :], y)
x = np.linspace(0, 1, 11)
plt.plot(x, theta_init[0, 1] * x + theta_init[0, 0], c='g')
plt.plot(x, theta[0, 1] * x + theta[0, 0], c='r')
plt.legend(["Initial", "Optimized"])

plt.show()