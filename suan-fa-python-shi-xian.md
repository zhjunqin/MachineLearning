# 感知机算法python实现

算法的原始形式

```
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, input_x, input_y, learn_rate=1, algorithm='sgd'):
        self._input_x = input_x
        self._input_y = input_y
        self._rate = learn_rate
        self._algorithm = algorithm
        self._speparte_input()

    def _speparte_input(self):
        total = len(self._input_y)
        self._positive_x = []
        self._nagtive_x = []

        for i in range(total):
            if self._input_y[i] >= 0:
                self._positive_x.append(self._input_x[i])
            else:
                self._nagtive_x.append(self._input_x[i])

    def draw_input_data(self):
        x1 = [x[0] for x in self._positive_x]
        x2 = [x[1] for x in self._positive_x]
        plt.scatter(x1, x2, label='positive', color='g', s=30, marker="o")
        x1 = [x[0] for x in self._nagtive_x]
        x2 = [x[1] for x in self._nagtive_x]
        plt.scatter(x1, x2, label='nagtive', color='r', s=30, marker="x")
        #plt.grid(True)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Perceptron')
        plt.legend()
        plt.show()
        
input_x = [[3,5], [4,4], [4,5], [5,3.5], [5,2.5], [1.5,2.5], [2,3], [2,2], [3,2.5], [3,1]]
input_y = [1,1,1,1,1,-1,-1,-1,-1,-1]

pla = Perceptron(input_x, input_y)
pla.draw_input_data()
```

算法的对偶形式

训练数据集

