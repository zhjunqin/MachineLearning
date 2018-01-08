## 二项Logistic回归模型

二项Logistic回归模型（binomial logistic regression model）是一种分类模型，有条件概率分布$$P(Y|X)$$表示，形式为参数化的logistic分布。

#### 一、模型定义

模型是如下的条件概率分布：


$$
P(Y=1|X)=\dfrac{e^{w\cdot x+b}}{1+e^{w\cdot x+b}}
$$



$$
P(Y=0|X)=1-P(Y=1|X)=\dfrac{1}{1+e^{w\cdot x+b}}
$$


这里$$x\in R^n$$，$$Y\in {0, 1}$$，$$w \in R^n$$和$$b\in R$$是参数，$$w$$称为权值，$$b$$称为偏置。

给定输入实例$$x$$计算得到$$P(Y=1|x)$$和$$P(Y=0|x)$$，然后比较两个条件概率的大小，将实例$$x$$分到概率值较大的那一类。

为了方便，将权值向量和输入向量加以扩展，即$$w=(w_1, w_2, ..., w_n, b)^T$$，$$x=(x_1, x_2, ..., x_n, 1)^T$$。这样，上面的模型变成：


$$
P(Y=1|X)=\dfrac{e^{w\cdot x}}{1+e^{w\cdot x}}
$$



$$
P(Y=0|X)=1-P(Y=1|X)=\dfrac{1}{1+e^{w\cdot x}}
$$


### 二、发生比

在统计和概率理论中，一个事件或者一个陈述的发生比（英语：Odds）是该事件发生和不发生的比率，公式为：


$$
odds(p)=\dfrac{p}{1-p}
$$


其中$$p$$是该事件发生的概率，$$odds(p)$$是关于$$p$$的递增函数。

例如，如果一个人随机选择一星期7天中的一天，选择星期日的发生比是： $$\dfrac{1/7}{1-1/7}=1/6$$。不选择星期日的发生比是 $$6/1$$。

对odds取对数\(成为log of odds\)，也就是$$log\dfrac{p}{1-p}$$，称为对数几率，这个在正式的数学文献中会记为$$logit(p)$$，即：


$$
logit(p)=log\dfrac{p}{1-p}
$$


该函数还是关于$$p$$的递增函数。

在Logistic回归中，对于某个实例$$x$$：


$$
log\dfrac{p}{1-p}=log\dfrac{P(Y=1|x)}{1-P(Y=1|x)}=w\cdot x
$$


也就是实例$$x $$输出$$Y=1$$的对数几率是$$x $$的线性函数。

### 三、极大似然估计

给定训练数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(N)},y^{(N)})\}$$，其中，$$x^{(i)}\in X= R^n$$，$$y^{(i)}\in Y=\{0, 1\}$$，应用极大似然估计发估计模型参数，从而得到Logistic回归模型。

设：$$P(Y=1|x)=\pi(x)=\dfrac{e^{w\cdot x}}{1+e^{w\cdot x}}$$，$$P(Y=0|x)=1-\pi(x)=\dfrac{1}{1+e^{w\cdot x}}$$

则似然函数为：


$$
\displaystyle\prod_{i=1}^N[\pi(x^{(i)})]^{y^{(i)}}[1-\pi(x^{(i)})]^{1-y^{(i)}}
$$


对数似然函数为：


$$
L(w)=\displaystyle\sum_{i=1}^N[y^{(i)}log\pi(x^{(i)})+(1-y^{(i)})log(1-\pi(x^{(i)}))]
$$



$$
=\displaystyle\sum_{i=1}^N[y^{(i)}log\dfrac{\pi(x^{(i)})}{1-\pi(x^{(i)})}+log(1-\pi(x^{(i)}))]
$$



$$
=\displaystyle\sum_{i=1}^N[y^{(i)}(w\cdot x^{(i)})-log(1+e^{w\cdot x^{(i)}})]
$$


该函数是高阶可导函数，对$$L(w)$$求极大值，即令每个样本的概率越大越好，得到$$w$$的估计值。

这样问题就变成了以对数似然函数为目标函数的最优化问题，Logistic回归学习中通常采用的方法是梯度下降和拟牛顿法。

### 四、交叉熵错误

模型中的概率，也可以表示成：


$$
P(Y=1|X)=\dfrac{e^{w\cdot x}}{1+e^{w\cdot x}}=\theta(w\cdot x)
$$


其中


$$
\theta(x)=\dfrac{e^{x}}{1+e^{x}}=\dfrac{1}{1+e^{-x}}
$$


于是极大似然函数可以写成：


$$
\max_{w}L(w)=ln\displaystyle\prod_{i=1}^N\theta(y^{(i)}w\cdot x^{(i)})
$$


改成取极小值：


$$
\min_{w}L(w)=-ln\displaystyle\sum_{i=1}^N\theta(y^{(i)}w\cdot x^{(i)})
$$



$$
=\displaystyle\sum_{i=1}^Nln(1+e^{-y^{(i)}w\cdot x^{(i)}})
$$


其中


$$
error(w,x,y)=ln(1+e^{-yw\cdot x})
$$


称为交叉熵错误。

### 五、极大似然函数的梯度

然后极大似然函数计算梯度：


$$
\dfrac{\partial L(w)}{\partial w_j}=\dfrac{\partial \displaystyle\sum_{i=1}^Nln(1+e^{-y^{(i)}w\cdot x^{(i)}})}{\partial w_j}
$$



$$
=\displaystyle\sum_{i=1}^N\dfrac{\partial ln(1+e^{-y^{(i)}w\cdot x^{(i)}})}{\partial w_j}
$$



$$
=\displaystyle\sum_{i=1}^N\big(\dfrac{1}{1+e^{-y^{(i)}w\cdot x^{(i)}}}\big) \big( e^{-y^{(i)}w\cdot x^{(i)}}     \big) \big(\dfrac{\partial -y^{(i)}w\cdot x^{(i)}}{\partial w_j}\big)
$$



$$
=\displaystyle\sum_{i=1}^N\big(\dfrac{e^{-y^{(i)}w\cdot x^{(i)}}}{1+e^{-y^{(i)}w\cdot x^{(i)}}}\big) \big(\dfrac{\partial -y^{(i)}w\cdot x^{(i)}}{\partial w_j}\big)
$$



$$
=\displaystyle\sum_{i=1}^N\big(\dfrac{e^{-y^{(i)}w\cdot x^{(i)}}}{1+e^{-y^{(i)}w\cdot x^{(i)}}}\big) \big(-y^{(i)}x^{(i)}_j\big)
$$

$$
=\displaystyle\sum_{i=1}^N\theta(-y^{(i)}w\cdot x^{(i)})(-y^{(i)}x^{(i)}_j)
$$


> 参考：
>
> 林轩田：机器学习基石
>
> [https://zh.wikipedia.org/wiki/发生比](https://zh.wikipedia.org/wiki/发生比)
>
> [http://vividfree.github.io/机器学习/2015/12/13/understanding-logistic-regression-using-odds](http://vividfree.github.io/机器学习/2015/12/13/understanding-logistic-regression-using-odds)



