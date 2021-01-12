import numpy as np
import matplotlib.pyplot as plt

X = np.array([np.ones(100), np.random.rand(100)])
y = np.dot([4, 3], X) + np.random.rand(100)
plt.scatter(X[1, :], y)
num_iters = 5

def newton(theta, X, y, num_iters):
    loss_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters, 2))
    m = len(y)
    theta = theta - (np.dot(theta, X) - y).dot(X.T).dot(np.linalg.inv(np.dot(X, X.T)))
    loss = 1 / (2 * m) * np.sum(np.square(np.dot(theta, X) - y))
    print("theta:{}".format(theta))
    print("loss:{}\n".format(loss))
    return theta, theta_history, loss_history

theta_init = np.random.randn(2)
theta, theta_history, loss_history = newton(theta_init, X, y, num_iters)
plt.scatter(X[1, :], y)

x = np.linspace(0, 1, 11)
plt.plot(x, theta_init[1] * x + theta_init[0], c='g')
plt.plot(x, theta[1] * x + theta[0], c='r')
plt.legend(["Initial", "Optimized"])
plt.show()
