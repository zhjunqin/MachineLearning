python实现

基本思路：构建一个树形结构的输出，非叶子节点都是特征，用特征将数据集分成

1首先计算



```
# -*- coding: utf-8 -*-

from math import log
from collections import Counter

class DecisionTree(object):
    def __init__(self, input_data, labels):
        pass
    
    def create_decision_tree(self, data_set, labels):
        class_list = [data[-1] for data in data_set]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(data_set[0]) == 1:
            major_label = Counter(data_set).most_common(1)[0]
            return major_label

        feature_index = self.get_feature_with_biggest_gain(data_set, labels) #获取最大信息增益的特征
        feature_name = labels[feature_index]
        del labels[feature_index]

        feature_set = set([ data[feature_index] for data in data_set])
        decision_tree = {feature_name: {}}
        for i in feature_set:
            feature_data_list = [ data for data in data_set if data[feature_index] == i ]
            new_data_list = []
            for j in feature_data_list:
                new_data = j[:]
                del new_data[feature_index]
                new_data_list.append(new_data)
            #print(i, new_data_list)
            new_lables = labels[:]
            decision_tree[feature_name][i] = self.create_decision_tree(new_data_list, new_lables)
        
        return decision_tree

    def cal_data_set_entropy(self, data_set):
        total_num = len(data_set)
        class_list = [data[-1] for data in data_set]
        class_dict = dict()
        for i in class_list:
            ck_num = class_dict.get(i, 0)
            class_dict[i] = ck_num + 1
        
        entropy = 0
        for k in class_dict:
            ck_rate = float(class_dict[k])/total_num
            entropy -= ck_rate * log(ck_rate, 2)
        return entropy

    def get_feature_with_biggest_gain(self, data_set, labels): #获取最大信息增益的特征
        feature_num = len(labels)
        data_entropy = self.cal_data_set_entropy(data_set)
        biggest_gain_index = None
        biggest_gain = 0
        for i in range(feature_num):
            condition_entroy = self.cal_feature_condition_entropy(data_set, i)
            gain = data_entropy - condition_entroy
            if gain > biggest_gain:
                biggest_gain_index = i
                biggest_gain = gain
        #print(labels[biggest_gain_index], biggest_gain)
        return biggest_gain_index

    def cal_feature_condition_entropy(self, data_set, index):
        total_num = len(data_set)
        feature_list = [data[index] for data in data_set]
        feature_dict = dict()
        for i in feature_list:
            feature_num = feature_dict.get(i, 0)
            feature_dict[i] = feature_num + 1

        condition_entropy = 0
        for k in feature_dict:
            feature_rate = float(feature_dict[k])/total_num
            feature_data_set = [data for data in data_set if data[index] == k]
            entropy = self.cal_data_set_entropy(feature_data_set)
            condition_entropy += feature_rate * entropy
        return condition_entropy
       



train_data = "/mnt/D/Learning/machineLearning/MachineLearning-python-2.7/input/3.DecisionTree/lenses.txt"
with open(train_data) as f:
    lenses = [line.strip().split('\t') for line in f.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
#[['young', 'myope', 'no', 'reduced', 'no lenses'], 
# ['young', 'myope', 'no', 'normal',  'soft']
dTree = DecisionTree(lenses, labels)
print(dTree.create_decision_tree(lenses, labels))

```

```
{'tearRate': {'reduced': 'no lenses', 
              'normal': {'astigmatic': {'yes': {'prescript': {'hyper': {'age': {'pre': 'no lenses', 
                                                                                'presbyopic': 'no lenses', 
                                                                                'young': 'hard'}}, 
                                                              'myope': 'hard'}}, 
                                        'no': {'age': {'pre': 'soft', 
                                                       'presbyopic': {'prescript': {'hyper': 'soft', 
                                                                      'myope': 'no lenses'}}, 
                                                       'young': 'soft'}}
                                        }
                        }
             }
}
```



