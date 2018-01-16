### 线性回归python实现

#### 1.算法python代码

包含Normal Equations，批量梯度下降和随机梯度下降，这里的代码跟Logistic回归的代码类似

```
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class LinearRegression(object):

    def __init__(self):
        self._history_w = []
        self._cost = []

    def load_input_data(self, data_file):
        with open(data_file) as f:
            input_x = []
            input_y = []
            for line in f:
                [x0, x1, y] = line.split()
                input_x.append([float(x0), float(x1)])
                input_y.append(float(y))
        self._input_x = np.array(input_x)
        self._input_y = np.array(input_y).T

    def normal_equations(self):  # 用矩阵的计算直接得到w
        xtx = np.dot(self._input_x.T, self._input_x)
        xtx_inverse = np.linalg.inv(xtx)
        tmp = np.dot(xtx_inverse, self._input_x.T)
        self._final_w = np.inner(tmp, self._input_y) # (X^T * X)^-1 * X^T * Y
        return self._final_w

    def cost_function(self, w):  # 成本函数
        tmp = np.inner(self._input_x, w)
        tmp = tmp - self._input_y
        return np.inner(tmp.T, tmp)

    def batch_gradient_descent(self, iter_num, iter_rate): #批量梯度下降
        (data_num, features) = np.shape(self._input_x)
        w = np.ones(features)
        for i in range(iter_num):
            inner = np.inner(self._input_x, w)
            delta = inner - self._input_y
            w = w - iter_rate * np.inner(self._input_x.T, delta) #w的迭代
            self._history_w.append(w)
            self._cost.append(self.cost_function(w))
        self._final_w = w
        return w

    def stochastic_gradient_descent(self, iter_num, iter_rate): # 随机梯度下降
        (data_num, features) = np.shape(self._input_x)
        w = np.ones(features)
        data_range = range(data_num)
        for i in range(iter_num):
            for j in data_range:
                iter_rate = 4/(1.0+j+i) + 0.01
                inner = np.inner(self._input_x[j], w)
                delta = inner - self._input_y[j]
                w = w - iter_rate * delta* self._input_x[j] #w的迭代
                self._history_w.append(w)
                self._cost.append(self.cost_function(w))
        self._final_w = w
        return w
```



