### 反向传播算法代码

给定一个大小为$$m$$的小批量数据，在小批量数据的基础上应用梯度下降学习算法：

1. **输入训练样本的集合**
2. **对于每个训练样本**$$x$$：设置对应的输入激活$$a^{x,1}$$，并执行下面的步骤  
   1. **前向传播**：对于每一个$$l=2,3,...,L$$，计算$$z^{x,l}=w^l a^{x,l-1} + b^l$$和$$a^{x,l}=\sigma(z^{x,l})$$

   1. **输出误差**$$\delta^{x,L}$$：计算向量$$\delta^{x,L}=\nabla_a C_x \odot \sigma'(z^{x,L})$$

   2. **反向传播误差**：对于每一个$$l=L-1,L-2,...,2$$计算$$\delta^{x,l}=((w^{l+1})^T \delta^{x, l+1}) \odot \sigma'(z^{x,l})$$

3. **梯度下降**：对于每一个$$l=L-1, L-2, ..., 2$$根据$$w^l\to w^l - \frac{\eta}{m}\displaystyle\sum_{x}\delta^{x,l} (a^{x,l-1})^T$$和$$b^l\to b^l - \frac{\eta}{m}\displaystyle\sum_{x}\delta^{x,l} $$更新权重和偏置。

#### 二次代价函数及sigmoid函数

如果使用二次代价函数，则每个训练样本的成本函数是$$C_x=\frac{1}{2}||y(x)-a^L(x)||^2$$，则$$\nabla_a C_x = y(x)-a(x)$$

sigmoid 激活函数$$\sigma(z) = \frac{1}{1+e^{-z}}$$的导数$$\sigma'(z) = \frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})=\sigma(z) (1-\sigma(z))$$

### Python 代码

这里的代码来自 [https://github.com/mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning)

```py
"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    # training_data 是一个 list 包含 5000 个元素，每个元素是一个 tuple (x, y)，其中
    # x 是一个 784 * 1 的向量
    # y 是一个 10 * 1 的向量，向量中某个位置为非 0 表示的是0-9的某个数字
    # validation_data 和 test_data 都是 list 包含 1000 个元素，每个元素是一个 tuple (x, y)，其中
    # x 是一个 784 * 1 的向量
    # y 是一个 0-9 的数字
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
```

    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    """
    network.py
    ~~~~~~~~~~

    A module to implement the stochastic gradient descent learning
    algorithm for a feedforward neural network.  Gradients are calculated
    using backpropagation.  Note that I have focused on making the code
    simple, easily readable, and easily modifiable.  It is not optimized,
    and omits many desirable features.
    """

    #### Libraries
    # Standard library
    import random

    # Third-party libraries
    import numpy as np

    class Network(object):

        def __init__(self, sizes):
            """The list ``sizes`` contains the number of neurons in the
            respective layers of the network.  For example, if the list
            was [2, 3, 1] then it would be a three-layer network, with the
            first layer containing 2 neurons, the second layer 3 neurons,
            and the third layer 1 neuron.  The biases and weights for the
            network are initialized randomly, using a Gaussian
            distribution with mean 0, and variance 1.  Note that the first
            layer is assumed to be an input layer, and by convention we
            won't set any biases for those neurons, since biases are only
            ever used in computing the outputs from later layers."""
            # size 是一个 list ，该长度表示了神经网络的总层数
            # list 里面每一个元素的值表示的是神经元的个数
            # 举例来说，输入 size=[784, 30, 10]，则神经网络层数有3层
            # 第一层有 784 个神经元，第二层有 30 个，第三层有 10 个
            self.num_layers = len(sizes)
            self.sizes = sizes
            # biases 是一个 list，每个元素是 (n * 1) 的向量，表示的神经网络对应层的偏置个数
            # 第一层输入，所以偏置从第二层开始
            # 举例来说，输入 size=[784, 30, 10]
            # 则，biases list 有两个元素，第一个元素是 (30 * 1 )，第二个元素是 (10 * 1)
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            # weights 是一个list， 每个元素是 (n * m)的矩阵，
            # 其中 n 是下一层网络的神经元个数，m 是上一层网络的神经元个数
            # 举例来说，输入 size=[784, 30, 10]
            # 则， weights 有两个元素，第一个元素是 (30 * 784)的矩阵，第二个元素是 (10 * 30)的矩阵
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        def feedforward(self, a):
            """Return the output of the network if ``a`` is input."""
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a)+b)
            return a

        def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
            """Train the neural network using mini-batch stochastic
            gradient descent.  The ``training_data`` is a list of tuples
            ``(x, y)`` representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If ``test_data`` is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
            if test_data: n_test = len(test_data)
            n = len(training_data)
            for j in xrange(epochs):
                random.shuffle(training_data)
                mini_batches = [training_data[k:k+mini_batch_size] 
                                for k in xrange(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
                else:
                    print "Epoch {0} complete".format(j)

        def update_mini_batch(self, mini_batch, eta):
            """Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
            is the learning rate."""
            # 初始化梯度值矩阵为 0
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

        def backprop(self, x, y):
            """
            返回一个 tuple，里面包含 (nabla_b, nabla_w)
            其中 nabla_b 是一个 list，每一个元素又是一个 arrays，包含的是网络中每一个 Layer 的 b 梯度值
            其中 nabla_w 是一个 list，每一个元素又是一个 arrays，包含的是网络中每一个 Layer 的 w 梯度值
            Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``.
            """
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            # 通过前向传播，计算神经网络所有层的线性输入值和激活输出
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            # 计算最后一层的输出误差
            # cost_derivative 是二次代价函数的导数
            # sigmoid_prime 是 sigmode 激活函数的导数
            # * 运算是按照每个元素的 Hadamard 乘积
            # 结果的 delta 是一个向量，大小为 (最后一层的神经元个数, 1)
            delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            # 偏置的改变率和偏差值相同
            nabla_b[-1] = delta
            # 权重的改变率是下一层的偏差和前一层的激活值向量內积
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                # 计算第 l-1 层的误差向量
                # 权重矩阵和误差的矩阵乘积
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)

        def evaluate(self, test_data):
            """Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation."""
            # np.argmax 获取到神经网络输出的向量中最大值的 index，也就是 0-9
            test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
            # 计算所有输出和测试数据相同的样本数量
            return sum(int(x == y) for (x, y) in test_results)

        def cost_derivative(self, output_activations, y):
            """Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations."""
            # 二次代价函数的导数
            return (output_activations-y)

    #### Miscellaneous functions
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))

执行算法：

```
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])
from datetime import datetime as datetime
print(datetime.now())
net.SGD(training_data, 30, 10, 3.0, test_data=test_data);
print(datetime.now())
```

输出：

```
Epoch 0: 9040 / 10000
Epoch 1: 9156 / 10000
Epoch 2: 9256 / 10000
...
```



