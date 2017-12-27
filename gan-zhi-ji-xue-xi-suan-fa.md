# 感知机学习算法

感知机学习问题转化为求解损失函数的最优化问题，最优化的方法就是随机梯度下降法。

### 1. 学习算法的原始形式

给定一个训练数据集$$T=\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$$，其中，$$x_i\in X= R^n$$，$$y_i\in Y=\lbrace+1,-1\rbrace$$，$$i=1,2,...,n$$，求参数$$w$$，$$b$$，使得其为以下损失函数极小化的解：


$$
\min_{w,b}L(w,b)=-\displaystyle\sum_{x_i\in M}y_i(w\cdot x_i+b)
$$
其中$$M$$为**误分类点**的集合。

假设误分类点集合$$M$$M时固定的，那么损失函数$$L(w,b)$$的梯度由


$$
\nabla_w L(w,b)=-\displaystyle\sum_{x_i\in M}y_i x_i
$$

$$
\nabla_b L(w,b)=-\displaystyle\sum_{x_i\in M}y_i
$$
给出。

