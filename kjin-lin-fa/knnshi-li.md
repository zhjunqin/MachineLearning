### KNN实例

本例来自《机器学习实战》，采用直接算两点之间距离的方式来找出最近邻

判断手写的数字，数据被转成32x32（=128）的数字矩阵

```
00000000000001100000000000000000
00000000000011111100000000000000
00000000000111111111000000000000
00000000011111111111000000000000
00000001111111111111100000000000
00000000111111100011110000000000
00000001111110000001110000000000
00000001111110000001110000000000
00000011111100000001110000000000
00000011111100000001111000000000
00000011111100000000011100000000
00000011111100000000011100000000
00000011111000000000001110000000
00000011111000000000001110000000
00000001111100000000000111000000
00000001111100000000000111000000
00000001111100000000000111000000
00000011111000000000000111000000
00000011111000000000000111000000
00000000111100000000000011100000
00000000111100000000000111100000
00000000111100000000000111100000
00000000111100000000001111100000
00000000011110000000000111110000
00000000011111000000001111100000
00000000011111000000011111100000
00000000011111000000111111000000
00000000011111100011111111000000
00000000000111111111111110000000
00000000000111111111111100000000
00000000000011111111110000000000
00000000000000111110000000000000
```

数据源 https://github.com/apachecn/MachineLearning/tree/python-2.7/input/2.KNN

数据源被分成两个部分，一部分是训练数据，一部分是测试数据

```
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import operator as op
import os

class KNN(object):

    def get_k_nearest(self, input_x, input_y, target, k): 
        # 计算目标跟所有数据的距离，找出最近的几个，并将数量最多的标签输出
        nearest = []
        (data_num, axes_num) = np.shape(input_x)
        repeat_target = np.tile(target, (data_num, 1)) # 将目标向量扩展成数据集一样数量的矩阵
        diff_matrix = repeat_target - input_x # 相减后，矩阵中的每个元素为 x1-y1
        square = np.multiply(diff_matrix, diff_matrix) # 每个元素做平方
        distance = square.sum(axis=1) # 计算距离 (x1-x21)^2 + x
        sorted_distance = distance.argsort() # 排序
        class_count = {}
        for i in range(k):
            lable = input_y[sorted_distance[i]]
            class_count[lable] = class_count.get(lable, 0) + 1
        sorted_class = sorted(class_count.iteritems(), 
                              key=op.itemgetter(1), # 依照第二个元素排序
                              reverse=True)  # 找出k个中最大
        return sorted_class[0][0]

    def digit_test(self, train_dir, test_dir): # 查找近似的数字
        def img_to_vector(file_name): # 将单个数字的数据存为向量
            return_vector = np.zeros(1024)
            f = open(file_name)
            sequence = range(32)
            for i in sequence:
                line = f.readline()
                for j in sequence:
                    return_vector[32*i + j] = int(line[j])
            return return_vector

        input_y = []
        input_x = []
        train_files = os.listdir(train_dir) 
        for i in train_files: # 将所有文件的数据都存为矩阵
            lable = int(i.split('_')[0])
            input_y.append(lable)
            input_x.append(img_to_vector('{}/{}'.format(train_dir,i)))
        
        input_x = np.array(input_x)
        input_y = np.array(input_y)
        test_files = os.listdir(test_dir)
        error = 0
        for i in test_files:
            lable = int(i.split('_')[0])
            test_vector = img_to_vector('{}/{}'.format(test_dir,i))
            predict_lable = self.get_k_nearest(input_x, input_y, test_vector, 3) # 执行knn
            #print("origin {}, predict {}".format(lable, predict_lable))
            if predict_lable == lable:
                continue
            error += 1
        print("error number is {}".format(error))

knn = KNN()
train_dir = "path/to/input/2.KNN/trainingDigits/"
test_dir = "path/to/input/2.KNN/testDigits/"
knn.digit_test(train_dir, test_dir)
```

结果：

```
error number is 12
```



