
import numpy as np

import time
import numpy as np

import time
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim

import  torchvision
from    matplotlib import pyplot as plt

batch_size = 512

# step1. load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
# torch.Size([512, 1, 28, 28]) torch.Size([512])
x_train = np.array(x.reshape(512,784).T[:,0])


y = one_hot(y)
y_train = np.array(y)

x_val = np.array(x.reshape(512,784).T[:,0])
y_val = np.array(y)

x_train = []
y_train = []
for i in range (5):
    x_train.append(np.random.rand(784,1))
    y_train.append(np.random.rand(10,1))
x_val = x_train.copy()
y_val = y_train.copy()




class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['O0'] = x_train
        # input layer to hidden layer 1
        params['X1'] = np.dot(params["W1"], params['O0'])
        params['O1'] = self.sigmoid(params['X1'])

        # hidden layer 1 to hidden layer 2
        params['X2'] = np.dot(params["W2"], params['O1'])
        params['O2'] = self.sigmoid(params['X2'])

        # hidden layer 2 to output layer
        params['X3'] = np.dot(params["W3"], params['O2'])
        params['O3'] = self.softmax(params['X3'])

        return params['O3']

    def backward_pass(self, y_train, output):

        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['X3'], derivative=True)
        change_w['W3'] = np.outer(error, params['O2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['X2'], derivative=True)
        change_w['W2'] = np.outer(error, params['O1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['X1'], derivative=True)
        change_w['W1'] = np.outer(error, params['O0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
         for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):

        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration + 1, time.time() - start_time, accuracy * 100
            ))


dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_val, y_val)