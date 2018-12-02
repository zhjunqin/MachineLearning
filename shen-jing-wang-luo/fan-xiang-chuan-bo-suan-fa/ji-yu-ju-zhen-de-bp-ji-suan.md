上面一章的 \`update\_mini\_batch\` 函数是每个数据集独立计算后向传播，后面再做平均值，

        def update_mini_batch(self, mini_batch, eta):
            """Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate."""
            # 初始化梯度值矩阵为 0
            # Nabla算子，在中文中也叫向量微分算子、劈形算子、倒三角算子
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                # 迭代计算梯度矩阵和
                # 获取当前样本通过反向传播算法得到的 delta 梯度值
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                # 把 mini_batch 里面每个数据算出来的梯度做加和，后面再取平均
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # 把梯度值取平均，并乘以系数 eta，然后更新权重和偏置矩阵
            self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

这里介绍一下如何通过矩阵的方式直接计算一个 mini batch 的梯度值向量。代码来源：https://github.com/hindSellouk/Matrix-Based-Backpropagation/blob/master/Network1.py

```
import random
from os import access

import numpy as np
import time

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            start_time = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("--- %s seconds elapsed---" % (time.time() - start_time))
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
    　　# mini batch 是一个 list，list 里面每个元素是一个 tuple (x, y)
        matrix_X=mini_batch[0][0]
        matrix_Y=mini_batch[0][1]
        #create matrix_X of examples and a matrix_Y of labels
        for x,y in mini_batch[1:]:
            # 将 mini batch 里面的每个数据拼接起来，成一个 matrix
            matrix_X = np.concatenate((matrix_X,x), axis=1)
            matrix_Y = np.concatenate((matrix_Y,y), axis=1)

        nabla_b, nabla_w = self.backprop(matrix_X,matrix_Y)
        # 下面将返回的权重和偏置梯度值做平均
        self.weights = [w - (eta / len(mini_batch)) * nw
                    for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                   for b, nb in zip(self.biases, nabla_b) ]


    def backprop(self, x, y):
        # 这里 x 和 y 就都是一个 matrix，包括一个 batch 大小的数据
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activation matrices, layer by layer
        zs = [] # list to store all the "sum of weighted inputs z" matrices, layer by layer
        i=0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 这里 delta 就是一个矩阵，如果 batch 为 10 大小的话
        # ('backprop delta', (10, 10), <type 'numpy.ndarray'>)
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        # 这里先将上面的 delta 矩阵做了加和，变成了
        #  (10,), <type 'numpy.ndarray'>)
        # 然后做了扩展，变成如下
        # ('nabla_b[-1]', (10, 1), <type 'numpy.ndarray'>)
        nabla_b[-1] = np.expand_dims(np.sum(delta,axis=1),axis=1)
        # 最后一层的权重矩阵
        # ('nabla_w[-1]', (10, 30), <type 'numpy.ndarray'>)
        # 这里 np.dot 将 batch 的所有权重做了线性和
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # 10 * 10
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # 这里 np.sum 将 batch 的所有误差做了线性和
            nabla_b[-l] = np.expand_dims(np.sum(delta,axis=1),axis=1)
            # 这里 np.dot 将 batch 的所有权重做了线性和
            # 原来 delta 是 n * 1, activations[-l - 1].transpose() 是  1 * m
            # batch 的情况下，delta 是 n * batch size, activations[-l - 1].transpose() 是 batch size * m
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
```

执行代码：

```
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network1
net = network1.Network([784, 30, 10])
from datetime import datetime as datetime
print(datetime.now())
net.SGD(training_data, 30, 10, 3.0, test_data=test_data);
print(datetime.now())
```

基于矩阵运算的算法一个 Epoch 大约为 3 秒:

```
--- 3.12179398537 seconds elapsed---
Epoch 0: 9061 / 10000
--- 3.03969407082 seconds elapsed---
Epoch 1: 9177 / 10000
--- 3.00676393509 seconds elapsed---
Epoch 2: 9261 / 10000

```

不基于矩阵运算的算法一个 Epoch 大约为 11 秒:

```
--- 11.0207378864 seconds elapsed---
Epoch 0: 9079 / 10000
--- 11.092361927 seconds elapsed---
Epoch 1: 9247 / 10000
--- 11.0678529739 seconds elapsed---
Epoch 2: 9326 / 10000
```



