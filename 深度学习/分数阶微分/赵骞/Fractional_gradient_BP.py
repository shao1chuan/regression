'''
在numpy_BP.py基础上，根据论文”Fractional-order deep bakcpropagation neural network"
修改成分数阶梯度下降
'''
import numpy as np
import random
from scipy.special import gamma #分数阶微分用

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class MLP_np:
    def __init__(self, sizes):
        '''
        :param sizes:[784, 30, 10]
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        # sizes: [784, 30, 10]
        # w: [ch_out, ch_in]
        # b: [ch_out]

        self.weights = [np.random.randn(ch2, ch1)  for ch1, ch2 in zip(sizes[:-1], sizes[1:])] #[30,784], [10,30]
        # z = wx +b [30, 1]
        self.biases = [np.random.randn(ch,1) for ch in sizes[1:]] #[30,1], [10,1]

        # 负数的小数次方会出现问题，所以要讲负数变为零
        #self.weights = list(map(lambda x: x.clip(0), self.weights))
        #self.biases = list(map(lambda x: x.clip(0), self.biases))

    def forward(self, x):
        '''
        :param x: [784,1]
        :return:
        '''
        for b, w in zip(self.biases, self.weights):
            # [30, 784] @ [784, 1] => [30, 1] + [30, 1] => [30,1]
            z = np.dot(w, x) + b
            # [30, 1]
            x = sigmoid(z)
        return x

    def backprop(self, x, y, der_order):
        '''
        :param x: [784, 1]
        :param y: [10,1], one_hot encoding
        :return:
        '''

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # 1. forward
        # save activation for every layer
        activations = [x] #存放各层节点输出值（即节点值，从原网络第一层开始）
        # save z for every layer， z是节点输入值
        zs = []  # zs[]存放各层节点输入值（从原网络第二层开始）
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) +b
            activation = sigmoid(z)

            zs.append(z) #节点输入值
            activations.append(activation) #节点输出值

        loss = np.power(activations[-1]-y, 2).sum()
        # 2. backward
        # 2.1 compute gradient on output layer
        # [10, 1] with [10, 1] => [10, 1] #书写数字0~9
        delta = activations[-1] * (1-activations[-1]) * (activations[-1] -y) # “loss” 对 “最后一层输入” 的 偏导数
        nabla_b[-1] = delta
        # [10, 1]@[1, 30]   =>[10, 30]
        # activation: [30, 1]
        nabla_w[-1] = np.dot(delta, activations[-2].T) #这个是重点********

        # 2.2 compute hidden gradient
        for l in range(2, self.num_layers):
            l = -l

            z = zs[l]
            a = activations[l]

            #delta_j
            # [10, 30]T@[10, 1] => [30, 10]@[10, 1] => [30, 1] * [30, 1] => [30, 1]
            delta = np.dot(self.weights[l+1].T, delta) *a * (1-a)

            nabla_b[l] = delta
            # [30, 1] @ [784, 1]T => [30, 784]
            nabla_w[l] = np.dot(delta, activations[l-1].T)
        return nabla_w, nabla_b, loss

    def train(self, training_data, epochs, batchsz, lr, test_data, der_order):
        '''
        :param training_data: list of (x,y)
        :param epochs: 1000
        :param batchsz: 10
        :param lr: 0.01
        :param test_data: list of (x,y)
        :param der_order：导数阶次
        :return:
        '''
        if test_data: n_test = len(test_data) #10000
        n = len(training_data) #50000
        for j in range(epochs): #1000
            random.shuffle(training_data) #列表中元素随机排序
            mini_batches = [training_data[k:k+batchsz] for k in range(0, n, batchsz)] #将training_data分割成很多小份儿，每份儿大小 bathchsz

            # for every sample in current batch
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, lr, der_order)  #每训练一个 batch 参数调整一次
            if test_data:
                print("Epoch {0}: {1} /{2}".format(
                    j, self.evaluate(test_data), n_test), loss)
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, lr, der_order): #意思是每运行一个 batch，参数更新一次
        '''
        :param batch: list of (x,y)
        :param lr: 0.01
        :param der_order: 导数阶次
        :return:
        '''
        # 建立一个同w，b同样结构的 nabla_w, nabla_b，用于存导数。（每运行一批都要新建一次这样的结构，是不是效率有些低？？？）
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = 0

        #for every sample in current batch
        for x, y in batch:
            # list of every w/b gradient
            # [w1, w2, w3]
            nabla_w_, nabla_b_, loss_ = self.backprop(x, y, der_order)
            #######################   根据论文 （18）式 ####################################

            nabla_w_ = list(map(lambda x,y: x * y**(1-der_order)/gamma(2-der_order), nabla_w_, self.weights))
            nabla_b_ = list(map(lambda x,y: x * y**(1-der_order)/gamma(2-der_order), nabla_b_, self.biases))#这句论文中没有，论文中没有 biases 项
            #################################################################################

            nabla_w = [accu+cur for accu, cur in zip(nabla_w, nabla_w_)]
            nabla_b = [accu+cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_

        nabla_w = [w/len(batch) for w in nabla_w] # 一个batch中，每次梯度的平均值
        nabla_b = [b/len(batch) for b in nabla_b]
        loss = loss/len(batch)

        # w = w - lr*nabla_w
        self.weights = [w - lr*nabla for w, nabla in zip(self.weights, nabla_w)] #这步是重点**********
        self.biases = [b - lr*nabla for b, nabla in zip(self.biases, nabla_b)]

        #负数的小数次方会出现问题，所以要讲负数变为零
        #self.weights = list(map(lambda x: x.clip(0), self.weights))
        #self.biases = list(map(lambda x: x.clip(0), self.biases))

        return loss

    def evaluate(self, test_data):
        '''
        y is not one-encoding.
        :param test_data: list of (x,y)
        :return:
        '''

        result = [(np.argmax(self.forward(x)), y) for x, y in test_data]
        correct = sum(int(pred==y) for pred, y in result)
        return correct

def main():
    import mnist_loader #需要 mnist_loader.py 和 data/mnist.pkl.gz文件
    # Loading the MNIST data
    # training_data: 50000行2列，列1:[784,1],列2:[10,1]（数字概率）
    # validation_data:10000行2列，列1:[784,1],列2:[0](手写数字值）
    # test_data:10000行2列，列1:[784,1], 列2:[0]（手写数字值）
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #print(len(training_data), training_data[0][0].shape, training_data[0][1].shape)
    #print(len(test_data), test_data[0][0].shape, test_data[0][1].shape)
    #print(test_data[0][1])

    #set up a Network with 30 hidden neurons
    net = MLP_np([784, 30, 10])

    net.train(training_data, 1000, 10, 0.01, test_data=test_data, der_order=1)

if __name__ == '__main__':
    main()





