### Softmax

除了使用交叉熵来解决学习缓慢的问题外，还可以使用基于柔性最大值（softmax）神经元层。

柔性最大值的想法其实就是为神经网络定义一种新式的输出层。开始时和和S型层一样，首先计算带权输入


$$
z^L_j=\displaystyle\sum_{k}(w_{jk}^L a_k^{L-1}) + b_j^L
$$


然后应用 softmax 函数在$$z^L_j$$上，根据这个函数，第$$j$$个神经元的激活值就是


$$
a^L_j = \frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}}
$$


其中分母的求和是在所有的输出神经元上进行的。而且所有输出的激活值加起来正好为1，同样保证输出激活值都是正数。而且柔性最大值层的输出可以被看做是一个概率分布。

下面计算$$a^L_j$$对$$z_i^L$$的导数

如果$$j=i$$：


$$
\frac{\partial a_j^L}{\partial z_i^L} = \frac{\partial }{\partial z_i^L}(\frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}})
$$



$$
=\frac{(e^{z^L_j})'\cdot\displaystyle\sum_{k}e^{z^L_k} - e^{z^L_j}\cdot e^{z^L_j} }{(\displaystyle\sum_{k}e^{z^L_k})^2}
$$



$$
=\frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}}-\frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}} \cdot \frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}}
$$



$$
=a_j (1-a_j)
$$


如果$$j \not= i$$：


$$
\frac{\partial a_j^L}{\partial z_i^L}=\frac{\partial }{\partial z_i^L}(\frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}})
$$



$$
=\frac{0\cdot\displaystyle\sum_{k}e^{z^L_k} - e^{z^L_j}\cdot e^{z^L_i} }{(\displaystyle\sum_{k}e^{z^L_k})^2}
$$



$$
=-\frac{e^{z^L_j}}{\displaystyle\sum_{k}e^{z^L_k}} \cdot \frac{e^{z^L_i}}{\displaystyle\sum_{k}e^{z^L_k}}
$$



$$
= - a_j a_i
$$


### 对数似然损失函数

其对数似然损失函数为：


$$
C=-\displaystyle\sum_{k}y_k \mathrm{log}a_k
$$
其中$$a_k$$为第$$k$$个神经元的输出值，$$y_k$$表示第$$k$$个神经元的真实值，取值为$$0$$或$$1$$。

这个代价的简单含义是：只有一个神经元对应了该样本的正确分类，若这个神经元的输出概率越高，则其产出的代价越小，反之则代价越高。

则计算损失函数对权重和偏置的偏导数：


$$
\frac{\partial C}{\partial b_j^L} = \frac{\partial C }{\partial z_j^L}\cdot \frac{\partial z_j^L }{\partial b_j^L}
$$



$$
= \frac{\partial C }{\partial z_j^L}\cdot \frac{\partial \displaystyle\sum_{k}(w_{jk}^L a_k^{L-1}) + b_j^L }{\partial b_j^L} = \frac{\partial C }{\partial z_j^L}
$$

$$
= \frac{\partial}{\partial z_j^L}( -\displaystyle\sum_{k}y_k \mathrm{log}a_k^L )
$$

$$
=-\displaystyle\sum_{k}y_k \cdot \frac{1 }{a_k^L} \cdot \frac{\partial a_k^L}{\partial z_j^L}
$$

$$
=-y_j \cdot \frac{1 }{a_j^L} \cdot \frac{\partial a_j^L}{\partial z_j^L}-\displaystyle\sum_{k\not=j}y_k \cdot \frac{1 }{a_k^L} \cdot \frac{\partial a_k^L}{\partial z_j^L}
$$

$$
=-y_j \cdot \frac{1 }{a_j^L} \cdot a_j^L (1-a_j^L)-\displaystyle\sum_{k\not=j}y_k \cdot \frac{1 }{a_k^L} \cdot -a_j^L a_k^L
$$

$$
=-y_j + y_j a_j^L +\displaystyle\sum_{k\not=j}y_k \cdot  a_j^L
$$

$$
=a_j^L-y_j
$$
同样可得：


$$
\frac{\partial C}{\partial w_{jk}}=a^{L-1}(a_j^L-y_j)
$$


因此可以确保不会遇到学习缓慢的问题。事实上把一个具有对数似然代价的柔性最大值输出层，看作与一个具有交叉熵代价函数的S型输出层非常相似。在很多应用场景中，这两种方式的效果都不错。

参考：https://blog.csdn.net/niuniuyuh/article/details/61926561

