### 线性回归模型（linear regression）

#### 1.模型定义

给定数据集，$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x^{(i)}=(1, x_1, x_2, ..., x_n)^T\in X= R^{n+1}$$，$$y^{(i)}\in Y=R$$，线性回归模型试图学到一个通过属性的线性组合来进行预测的函数，即


$$
f(x)=w_1x_1+w_2x_2+...+w_nx_n+b
$$


一般用向量写成：


$$
f(x)=w^T\cdot x+b
$$


其中$$w=(w_1, x_2, ..., w_n)^T\in R^{n}$$，$$b\in R$$，使得$$f(x)\simeq y$$。

如何确定$$w$$和$$b$$呢，显然关键在于如何衡量$$f(x)$$与$$y$$之间的差别。均方误差是最常用的性能度量，因此我们可以试图让均方误差最小化，即：


$$
\min_{w,b} L(w,b)=\displaystyle\sum_{i=1}^m(f(x^{(i)})-y^{(i)})^2
$$



$$
=\displaystyle\sum_{i=1}^m(w^T\cdot x^{(i)}+b-y^{(i)})^2
$$


均方误差有非常好的几何意义，它对应了常用的欧几里得距离或简称“**欧氏距离**”（Euclidean distance）。基于均方误差最小化来进行模型求解的方法称为“**最小二乘法**”（least square method）。在线性回归中，最小二乘法是试图找到一条直线，使得所有样本到直线上的欧氏距离之和最小。

求解$$w$$和$$b$$，使$$ L(w,b)=\displaystyle\sum_{i=1}^m(f(x^{(i)})-y^{(i)})^2$$最小化的过程，称为线性回归模型的最小二乘“参数估计”（parameter estimation）。令$$w_0=b$$，$$x_0=1$$，则$$w=(w_0,w_1, w_2, ..., w_n)^T$$，$$x=(x_0, x_1, x_2, ..., x_n)^T$$，原式转换为：


$$
f(x)=w^T\cdot x
$$



$$
\min_{w}  L(w)=\displaystyle\sum_{i=1}^m(w^T\cdot x^{(i)}-y^{(i)})^2
$$


对其求导，可得：


$$
\dfrac{\partial L(w,b)}{\partial w_j}=\dfrac{\partial \displaystyle\sum_{i=1}^m(w^T\cdot x^{(i)}-y^{(i)})^2}{\partial w_j}
$$



$$
=\displaystyle\sum_{i=1}^m2(w^T\cdot x^{(i)}-y^{(i)})x^{(i)}_j
$$


得到梯度向量：


$$
\nabla L(w)= \displaystyle\sum_{i=1}^m(w^T\cdot x^{(i)}-y^{(i)})x^{(i)}
$$


假定：

$$X= \begin{bmatrix}
   (x^{(1)})^T \\
   (x^{(2)})^T \\
   (x^{(3)})^T \\
     ... \\
   ( x^{(m)} )^T 
\end{bmatrix} = \begin{bmatrix}
   1 & x^{(1)}_1 & x^{(1)}_2 & ... & x^{(1)}_n \\
   1 & x^{(2)}_1 & x^{(2)}_2 & ... & x^{(2)}_n \\
   1 & x^{(3)}_1 & x^{(3)}_2 & ... & x^{(3)}_n \\
                                 ... \\
   1 & x^{(m)}_1 & x^{(m)}_2 & ... & x^{(m)}_n 
\end{bmatrix}$$，$$Y=\begin{bmatrix}
   y^{(1)} \\
   y^{(2)} \\
   y^{(3)} \\
        ... \\
   y^{(m)} 
\end{bmatrix}$$，$$w=\begin{bmatrix}
   w_0 \\
   w_1 \\
   w_2 \\
        ... \\
   w_n 
\end{bmatrix}$$

则：


$$
X\cdot w= \begin{bmatrix}
   1 & x^{(1)}_1 & x^{(1)}_2 & ... & x^{(1)}_n \\
   1 & x^{(2)}_1 & x^{(2)}_2 & ... & x^{(2)}_n \\
   1 & x^{(3)}_1 & x^{(3)}_2 & ... & x^{(3)}_n \\
                                 ... \\
   1 & x^{(m)}_1 & x^{(m)}_2 & ... & x^{(m)}_n 
\end{bmatrix}\cdot \begin{bmatrix}
   w_0 \\
   w_1 \\
   w_2 \\
        ... \\
   w_n 
\end{bmatrix}=\begin{bmatrix}
   (x^{(1)})^T\cdot w \\
   (x^{(2)})^T\cdot w \\
   (x^{(3)})^T\cdot w \\
                                 ... \\
   (x^{(m)})^T\cdot w 
\end{bmatrix}=\begin{bmatrix}
   w^T \cdot x^{(1)} \\
   w^T \cdot x^{(2)} \\
   w^T \cdot x^{(3)} \\
                                 ... \\
   w^T \cdot x^{(m)} 
\end{bmatrix}
$$



$$
X\cdot w-Y =\begin{bmatrix}
   w^T \cdot x^{(1)}-y^{(1)} \\
   w^T \cdot x^{(2)}-y^{(2)} \\
   w^T \cdot x^{(3)}-y^{(3)} \\
                                 ... \\
   w^T \cdot x^{(m)}-y^{(m)} 
\end{bmatrix}
$$



$$
X^T\cdot (X\cdot w-Y )=\begin{bmatrix}
   x^{(1)} & x^{(2)} & x^{(3)} & ... & x^{(m)} 
\end{bmatrix}\cdot \begin{bmatrix}
   w^T \cdot x^{(1)}-y^{(1)} \\
   w^T \cdot x^{(2)}-y^{(2)} \\
   w^T \cdot x^{(3)}-y^{(3)} \\
                                 ... \\
   w^T \cdot x^{(m)}-y^{(m)} 
\end{bmatrix}
$$


于是得到：


$$
\nabla L(w)=2 X^T\cdot (X\cdot w-Y )
$$


欲求的最优解，令上式为0，


$$
\nabla L(w)=2 X^T\cdot X\cdot w-2 X^T\cdot Y =0
$$


则得到：


$$
w=(X^T\cdot X)^{-1}\cdot X^T\cdot Y
$$


于是最终得到线性模型：


$$
f(x)=w^T\cdot x=x^T\cdot (X^T\cdot X)^{-1}\cdot X^T\cdot Y
$$


令$$X^\dagger =(X^T\cdot X)^{-1}\cdot X^T$$，称为**伪逆\(**seudo-inverse\)，代入得到。


$$
f(x) = x^T\cdot X^\dagger\cdot Y
$$


### 2.学习算法

2.1 矩阵计算

### 

#### 2.2 批量梯度下降（Batch Gradient Descent）

输入：训练数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x^{(i)}\in X= R^n$$，$$y^{(i)}\in Y=R^n$$，$$i=1,2,...,m$$，学习率$$\eta(0<\eta\leqslant1)$$；

输出：$$w=(w_1, w_2, ..., w_n)^T$$，，其中，模型

1）将输入的每个$$x_i$$转换成$$x_i=(1, x_1, x_2,...x_n)$$，令$$w_0=b$$，则输出为$$w^{(0)}=(w_0, w_1, w_2, ..., w_n)^T$$

2）选取初始$$w^{(0)}=(w_0, w_1, w_2, ..., w_n)^T$$

3）计算梯度$$x = y$$，其中为第次迭代的结果，则第次为：

4）转到步骤（3），一直到满足一定条件，或者迭代到足够的次数。

在批量梯度下降算法中，每一步的迭代都需要计算所有样本，当样本数较大时，计算量会很大。

时间复杂度：

每次迭代更新X⋅w\(j\)=Y′的计算次数为m×n，θ\(Y′\)−Y=Z的计算次数为n次，XT⋅Z的计算次数为m×n，则每次迭代的时间复杂度为O\(m×n\)，假定迭代次数为k次，则总时间复杂度为O\(k×m×n\).

#### 

#### 2.3 随机梯度下降（Stochastic Gradient Descent）

将上面的步骤（3）改为：

随机选取某个样本x\(i\)，则：

一直到迭代到足够的次数。

