### Regularization 规范化，正则化

#### L2 规范化

regularization（规范化，正则化）有时候被称为权重衰减（weight decay）

L2规范化的想法是增加一个额外的项到代价函数上，这个项叫做规范化项。


$$
C= -\frac{1}{n}\displaystyle\sum_{x}\displaystyle\sum_{y}[y_j\mathrm{ln}a^L_j+(1-y_j)\mathrm{ln}(1-a^L_j))] + \frac{\lambda}{2n} \displaystyle\sum_{w} w^2
$$


其中第一项就是常规的交叉熵d额表达式，第二项是所有权重的平方和。然后用一个因子$$\frac{\lambda}{2n}$$进行量化调整，其中$$\lambda$$称为规范化参数，而$$n$$就是训练集合的大小。需要注意的是，规范化项里面不包含偏置。

也可以对其他代价函数进行规范化，比如二次代价函数


$$
C= -\frac{1}{n}\displaystyle\sum_{x}||y-a^L||^2 + \frac{\lambda}{2n} \displaystyle\sum_{w} w^2
$$


两者都可以写成


$$
C= C_0 + \frac{\lambda}{2n} \displaystyle\sum_{w} w^2
$$


其中$$C_0$$就是原始的代价函数。

计算对网络中所有权重和偏置的偏导数
$$
\frac{\partial C}{\partial w }=\frac{\partial C_0}{\partial w}+\frac{\lambda}{n} w
$$

$$
\frac{\partial C}{\partial b }=\frac{\partial C_0}{\partial b}
$$
在反向传播中，梯度下降的偏置学习规则不会发生变化，权重的学习规则发生了变化。
$$
b \rightarrow b - \eta \frac{\partial C_0}{\partial b}
$$

$$
w \rightarrow w - \eta \frac{\partial C_0}{\partial w} -\eta \frac{\lambda}{n} w
$$

$$
= (1- \eta \frac{\lambda}{n} )w - \eta \frac{\partial C_0}{\partial w} 
$$
这个和通常的梯度下降学习规则相同，除了通过一个因子$$ (1- \eta \frac{\lambda}{n} )$$重新调整了权重，这种调整有时候被称为权重衰减，因为它使权重变小。

如果使用平均$$m$$个训练样本的小批量的数据来估计权重，则为了随机梯度下降的规范化学习规则就变成了
$$
w \rightarrow (1- \eta \frac{\lambda}{n} )w -\frac{ \eta}{m} \displaystyle\sum_{x} \frac{\partial C_x}{\partial w} 
$$

$$
b \rightarrow b - \frac{ \eta}{m} \displaystyle\sum_{x} \frac{\partial C_x}{\partial b} 
$$
其中后一项是在训练样本的小批量数据上进行的。

L1 规范化



