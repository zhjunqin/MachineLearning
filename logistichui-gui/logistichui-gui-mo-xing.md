## 二项Logistic回归模型

二项Logistic回归模型（binomial logistic regression model）是一种分类模型，有条件概率分布$$P(Y|X)$$表示，形式为参数化的logistic分布。

#### 一、模型定义

模型是如下的条件概率分布：


$$
P(Y=1|X)=\dfrac{e^{w\cdot x+b}}{1+e^{w\cdot x+b}}
$$



$$
P(Y=0|X)=\dfrac{1}{1+e^{w\cdot x+b}}
$$


这里$$x\in R^n$$，$$Y\in {0, 1}$$，$$w \in R^n$$和$$b\in R$$是参数，$$w$$称为权值，$$b$$称为偏置。

给定输入实例$$x$$计算得到$$P(Y=1|x)$$和$$P(Y=0|x)$$，然后比较两个条件概率的大小，将实例$$x$$分到概率值较大的那一类。

为了方便，将权值向量和输入向量加以扩展，即$$w=(w_1, w_2, ..., w_n, b)^T$$，$$x=(x_1, x_2, ..., x_n, 1)^T$$。这样，上面的模型变成：


$$
P(Y=1|X)=\dfrac{e^{w\cdot x}}{1+e^{w\cdot x}}
$$



$$
P(Y=0|X)=\dfrac{1}{1+e^{w\cdot x}}
$$






> 参考：
>
> http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/13/understanding-logistic-regression-using-odds



