### Logistic回归python实现

#### 1.算法python代码

```
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class Logistic(object):

    def __init__(self):
        self._history_w = []
        self._likelihood = []

    def load_input_data(self, data_file):
        with open(data_file) as f:
            input_x = []
            input_y = []
            for line in f:
                [x1, x2, y] = line.split()
                input_x.append([1.0, float(x1), float(x2)])
                input_y.append(int(y))
        self._input_x = np.array(input_x, dtype=np.float128)
        self._input_y = np.array(input_y, dtype=np.float128).T

    def sigmoid(self, x, w):
        return 1.0/(1+ np.exp(-np.inner(w,x)))

    def likelihood_function(self, w):
        temp = np.inner(self._input_x, w)
        a = np.inner(temp.T, self._input_y)
        b = np.sum(np.log(1+np.exp(temp)))
        return b-a

    def batch_gradient_descent(self, iter_num, iter_rate):
        (data_num, features) = np.shape(self._input_x)
        w = np.ones(features)
        for i in range(iter_num):
            theta = self.sigmoid(self._input_x, w)
            delta = theta - self._input_y
            w = w - iter_rate * np.inner(self._input_x.T, delta)
            self._history_w.append(w)
            self._likelihood.append(self.likelihood_function(w))
        self._final_w = w
        return w

    def stochastic_gradient_descent(self, iter_num, iter_rate):
        (data_num, features) = np.shape(self._input_x)
        w = np.ones(features)
        iter_range = range(iter_num)
        data_range = range(data_num)
        for i in range(iter_num):
            for j in data_range:
                iter_rate = 4/(1.0+j+i) + 0.01
                theta = self.sigmoid(self._input_x[j], w)
                delta = theta - self._input_y[j]
                w = w - iter_rate * delta* self._input_x[j]
                self._history_w.append(w)
                self._likelihood.append(self.likelihood_function(w))
        self._final_w = w
        return w
```



