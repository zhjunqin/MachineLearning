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

求解$$w$$和$$b$$使$$ L(w,b)=\displaystyle\sum_{i=1}^m(f(x^{(i)})-y^{(i)})^2$$最小化的过程，称为线性回归模型的最小二乘“参数估计”（parameter estimation）。令$$w_0=b$$，$$x_0=1$$，则$$w=(w_0,w_1, w_2, ..., w_n)^T$$，$$x=(x_0, x_1, x_2, ..., x_n)^T$$，原式转换为：


$$
f(x)=w^T\cdot x
$$

$$
\min_{w}  L(w)=\displaystyle\sum_{i=1}^m(w^T\cdot x^{(i)}-y^{(i)})^2
$$
对其求导，可得：


$$
\dfrac{\partial L(w,b)}{\partial w_j}=\dfrac{\partial \displaystyle\sum_{i=1}^m(w^T\cdot x^{(i)}-y^{(i)})^2}{\partial w_j}=\displaystyle\sum_{i=1}^m2(w^T\cdot x^{(i)}-y^{(i)})x^{(i)}_j
$$




