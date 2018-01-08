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


### 二、发生比

在统计和概率理论中，一个事件或者一个陈述的发生比（英语：Odds）是该事件发生和不发生的比率，公式为：


$$
odds(p)=\dfrac{p}{1-p}
$$


其中$$p$$是该事件发生的概率，$$odds(p)$$是关于$$p$$的递增函数。

例如，如果一个人随机选择一星期7天中的一天，选择星期日的发生比是： $$\dfrac{1/7}{1-1/7}=1/6$$。不选择星期日的发生比是 $$6/1$$。

对odds取对数\(成为log of odds\)，也就是$$log\dfrac{p}{1-p}$$，这个在正式的数学文献中会记为$$logit(p)$$，即：


$$
logit(p)=log\dfrac{p}{1-p}
$$
该函数还是关于$$p$$的递增函数。

> 参考：
>
> [https://zh.wikipedia.org/wiki/发生比](https://zh.wikipedia.org/wiki/发生比)
>
> [http://vividfree.github.io/机器学习/2015/12/13/understanding-logistic-regression-using-odds](http://vividfree.github.io/机器学习/2015/12/13/understanding-logistic-regression-using-odds)



