#### 朴素贝叶斯算法

给定数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x\in \mathcal{X}\subseteq R^n$$，$$y\in \mathcal{Y}=\{c_1, c_2,...,c_K\}$$，$$X$$是定义在输入空间$$\mathcal{X}$$上的随机向量，$$Y$$是定义在输出空间$$\mathcal{Y}$$上的随机变量，则对于输入$$x$$，计算如下的到输出$$ y$$。


$$
y=f(x)=\arg \max_{c_k}\prod_{j=1}^n P(X_j=x_j|Y=c_k)P(Y=c_k)
$$


式中的$$P(\cdot)$$值小于1，多个小于1的值连乘在python中执行会导致下溢，因此可以取对数，可以将乘法改为加法。而且对数函数是递增函数并不影响结果。则：


$$
y=f(x)=\arg \max_{c_k}ln\big(\prod_{j=1}^n P(X_j=x_j|Y=c_k)P(Y=c_k)\big)
$$



$$
=\arg \max_{c_k}\big(lnP(Y=c_k)+\displaystyle\sum_{i=1}^nln(P(X_j=x_j|Y=c_k)\big)
$$


#### 算法实现

> 数据源：[https://github.com/apachecn/MachineLearning/tree/python-2.7/input/4.NaiveBayes/email](https://github.com/apachecn/MachineLearning/tree/python-2.7/input/4.NaiveBayes/email)

例子里面中的基本步骤如下：

1. 将数据集切分称训练数据集和测试数据集。
2. 预先提取出所有的数据里面单词构成单词向量。
3. 然后分别将训练数据集和测试数据集的输入，分词，并转换称单词向量。
4. 然后进行训练，训练时计算各个单词的数量，然后除以总单词树，并使用lamda=1。
5. 然后进行测试，采样log的加和来使得避免连乘溢出。

```
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import random

class NativeBayes(object):

    def __init__(self):
        self._train_x = []  # 训练数据x
        self._train_y = []  # 训练数据y
        self._test_x = []   # 测试数据x
        self._test_y = []   # 测试数据y
        self._all_words = None # 所有的单词
        self._all_words_num = 0 # 所以单词数量

    def split_to_word(self, text): # 将文本切分称单词
        words = re.split(r'\W*', text)
        #return [word.lower() for word in words if len(word) > 2]
        return [word.lower() for word in words if len(word) > 2 and re.match(r'[a-zA-Z]', word)]

    def words_to_vector(self, words): # 将切分后的单词转换称单词向量
        vector = [0]*self._all_words_num
        for word in words:
            if word in self._all_words:
                vector[self._all_words.index(word)]  += 1
        return vector

    def load_data(self, positive_dir, nagetive_dir): 
        train_files = os.listdir(positive_dir)
        input_x = []
        input_y = []
        all_input_x = []
        for i in train_files:
            with open('{}/{}'.format(positive_dir,i)) as f:
                text = f.read()
                words = self.split_to_word(text)
                input_x.append(words)
                all_input_x.extend(words)
                input_y.append(1)

        train_files = os.listdir(nagetive_dir)
        for i in train_files:
            with open('{}/{}'.format(nagetive_dir,i)) as f:
                text = f.read()
                words = self.split_to_word(text)
                input_x.append(words)
                all_input_x.extend(words)
                input_y.append(-1)

        self._all_words = list(set(all_input_x)) # 获得数据里面所有的单词列表
        self._all_words_num = len(self._all_words) # 单词列表里面的单词数量

        total = len(input_y)
        test_x = []
        test_y = []
        for i in range(10):  # 将数据集分为训练数据和测试数据
            index = random.randint(0, total-1)
            test_x.append(input_x[index])
            test_y.append(input_y[index])
            del(input_x[index])
            del(input_y[index])
            total -= 1

        self._train_x = []
        self._train_y = input_y
        train_num = len(input_y)
        print('train data num', train_num)
        for i in range(train_num):  # 将训练数据单词列表转换称单词向量
            vector = self.words_to_vector(input_x[i])
            self._train_x.append(vector)

        self._test_x = []
        self._test_y = test_y
        test_num = len(test_y)
        print('test data num', test_num)
        for i in range(test_num): # 将测试数据单词列表转换称单词向量
            vector = self.words_to_vector(test_x[i])
            self._test_x.append(vector)

    def train(self):
        train_data_num = len(self._train_y)
        p_positive = np.ones(self._all_words_num) # 贝叶斯估计，所有单词初始化lamda=1
        p_negative = np.ones(self._all_words_num) # 贝叶斯估计，lamda=1
        positive_words_total = self._all_words_num # 同时所有的单词数量响应增加
        negative_words_total = self._all_words_num # 原书中此处为0，应该是错误的
        total_positive = 0
        for i in range(train_data_num):
            if self._train_y[i] == 1:
                p_positive += self._train_x[i]
                positive_words_total += sum(self._train_x[i])
                total_positive += 1
            else:
                p_negative += self._train_x[i]
                negative_words_total += sum(self._train_x[i])
        p_positive = np.log(p_positive/positive_words_total) # 计算各个单词的条件概率
        p_negative = np.log(p_negative/negative_words_total)
        positive_class = total_positive/float(train_data_num) # 计算分类概率
        print('train positive percent',positive_class)
        return p_positive,p_negative,positive_class

    def classify(self, p_positive, p_negative,  positive_class, vector):
        # 分别计算各个子类的概率
        positive = np.sum(p_positive*vector) + np.log(positive_class)
        nagative = np.sum(p_negative*vector) + np.log(1 - positive_class)
        print(positive,nagative)
        if positive > nagative:
            return 1
        else:
            return -1

    def test_data(self):
        p_positive,p_negative,positive_class = self.train()
        total_test = len(self._test_y)
        error_num = 0
        for i in range(total_test):
            vector = self._test_x[i]
            predict = self.classify(p_positive, p_negative, positive_class, vector)
            if predict != self._test_y[i]:
                error_num += 1
        print('predict error num', error_num)

bayes = NativeBayes()
postive = "/path_to/input/4.NaiveBayes/email/ham/"
nagative = "/path_to/input/4.NaiveBayes/email/spam/"
bayes.load_data(postive, nagative)
bayes.test_data()
```

测试结果：

```
('train data num', 40)
('test data num', 10)
('train positive percent', 0.5)
(-219.22174839643606, -225.29558562436705)
(-131.2753894915964, -150.43783388899158)
(-173.14800355833415, -180.89949078604346)
(-195.48666534835462, -193.29700989614457)
(-110.31437440519804, -93.453134060488026)
(-217.98906102300515, -156.13376765308851)
(-68.80147262518193, -76.076316158541744)
(-80.416051743878654, -65.808185492417721)
(-146.34615733070959, -156.19156326594504)
(-105.51819039117207, -114.69496509329458)
('predict error num', 1)
```



