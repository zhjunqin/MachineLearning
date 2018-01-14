## 二项Logistic回归模型

二项Logistic回归模型（binomial logistic regression model）是一种分类模型，由条件概率分布$$P(Y|X)$$表示，形式为参数化的logistic分布。

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

为了方便，将权值向量和输入向量加以扩展，即令$$w_0=b$$，$$x_0=1$$，扩展为


$$
w=(w_0,w_1, w_2, ..., w_n)^T
$$



$$
x=(x_0, x_1, x_2, ..., x_n)^T
$$


这样，上面的模型变成：


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

给定训练数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中，$$x^{(i)}=(1, x_1, x_2, ..., x_n)^T\in X= R^{n+1}$$，$$y^{(i)}\in Y=\{0, 1\}$$，应用极大似然估计发估计模型参数，从而得到Logistic回归模型。

设：$$P(Y=1|x)=\pi(x)=\dfrac{e^{w\cdot x}}{1+e^{w\cdot x}}$$，$$P(Y=0|x)=1-\pi(x)=\dfrac{1}{1+e^{w\cdot x}}$$

则似然函数为：


$$
\displaystyle\prod_{i=1}^m[\pi(x^{(i)})]^{y^{(i)}}[1-\pi(x^{(i)})]^{1-y^{(i)}}
$$


对数似然函数为：


$$
L(w)=\displaystyle\sum_{i=1}^m[y^{(i)}ln\pi(x^{(i)})+(1-y^{(i)})ln(1-\pi(x^{(i)}))]
$$



$$
=\displaystyle\sum_{i=1}^m[y^{(i)}ln\dfrac{\pi(x^{(i)})}{1-\pi(x^{(i)})}+ln(1-\pi(x^{(i)}))]
$$



$$
=\displaystyle\sum_{i=1}^m[y^{(i)}(w\cdot x^{(i)})-ln(1+e^{w\cdot x^{(i)}})]
$$


该函数是高阶可导函数，对$$L(w)$$求极大值，即令每个样本的概率越大越好，得到$$w$$的估计值。

变换成求极小值：


$$
\min_{w} L(w)=-\displaystyle\sum_{i=1}^m[y^{(i)}(w\cdot x^{(i)})-ln(1+e^{w\cdot x^{(i)}})]
$$


这样问题就变成了以对数似然函数为目标函数的最小值优化问题，Logistic回归学习中通常采用的方法是梯度下降和拟牛顿法。

计算梯度：


$$
\dfrac{\partial L(w)}{\partial w_j}=-\dfrac{\partial \displaystyle\sum_{i=1}^m[y^{(i)}(w\cdot x^{(i)})-ln(1+e^{w\cdot x^{(i)}})]}{\partial w_j}
$$



$$
= \displaystyle-\sum_{i=1}^m(y^{(i)}x^{(i)}_j)+\displaystyle\sum_{i=1}^m\dfrac{\partial ln(1+e^{w\cdot x^{(i)}})}{\partial w_j}
$$



$$
= \displaystyle-\sum_{i=1}^m(y^{(i)}x^{(i)}_j)+\displaystyle\sum_{i=1}^m\dfrac{1}{1+e^{w\cdot x^{(i)}}}\dfrac{\partial e^{w\cdot x^{(i)}}}{\partial w_j}
$$



$$
= \displaystyle-\sum_{i=1}^my^{(i)}x^{(i)}_j+\displaystyle\sum_{i=1}^m\dfrac{e^{w\cdot x^{(i)}}}{1+e^{w\cdot x^{(i)}}}x^{(i)}_j
$$



$$
= \displaystyle\sum_{i=1}^m\big(\dfrac{e^{w\cdot x^{(i)}}}{1+e^{w\cdot x^{(i)}}}-y^{(i)}\big)x^{(i)}_j
$$



$$
= \displaystyle\sum_{i=1}^m\big(\theta(w\cdot x^{(i)})-y^{(i)}\big)x^{(i)}_j
$$


其中$$\theta(x)=\dfrac{e^{x}}{1+e^{x}}=\dfrac{1}{1+e^{-x}}$$，也称为$$sigmoid$$函数，然后得到：


$$
\nabla L(w)= \displaystyle\sum_{i=1}^m\big(\theta(w\cdot x^{(i)})-y^{(i)}\big)x^{(i)}
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
\theta(X\cdot w)-Y=\begin{bmatrix}
   {\theta}(w^T \cdot x^{(1)})-y^{(1)} \\
   {\theta}(w^T \cdot x^{(2)})-y^{(2)} \\
   {\theta}(w^T \cdot x^{(3)})-y^{(3)} \\
                                 ... \\
   {\theta}(w^T \cdot x^{(m)})-y^{(m)} 
\end{bmatrix}
$$



$$
X^T= \begin{bmatrix}
   x^{(1)} & x^{(2)} & x^{(3)} & ... & x^{(m)} 
\end{bmatrix}
$$



$$
X^T\cdot \big(\theta(X\cdot w)-Y\big) = \displaystyle\sum_{i=1}^m\big(\theta(w\cdot x^{(i)})-y^{(i)}\big)x^{(i)}
$$


最终得到：


$$
\nabla L(w)= X^T\cdot \big(\theta(X\cdot w)-Y\big)
$$


同时也可以得到：


$$
 L(w)=-\displaystyle\sum_{i=1}^m[y^{(i)}(w\cdot x^{(i)})-ln(1+e^{w\cdot x^{(i)}})]=-(X\cdot w)^T\cdot Y+ln(1+e^{X\cdot w })\cdot I
$$


其中$$I$$为全$$1$$向量。

### 四、梯度下降法

#### 1.批量梯度下降（Batch Gradient Descent）

输入：训练数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x^{(i)}\in X= R^n$$，$$y^{(i)}\in Y=\lbrace0,1\rbrace$$，$$i=1,2,...,m$$，学习率$$\eta(0<\eta\leqslant1)$$；

输出：$$w$$，$$b$$，其中$$w=(w_1, w_2, ..., w_n)^T$$，模型$$P(y=1|x)=sigmoid ( w\cdot x+b)$$

1）将输入的每个$$x$$转换成$$x_i=(1, x_1, x_2,...x_n)$$，令$$w_0 =b$$，输出定为$$w=(w_0, w_1, w_2, ..., w_n)^T$$

2）选取初始$$w^{(0)}=(w_0, w_1, w_2, ..., w_n)^T$$

3）计算梯度$$X^T\cdot \big(\theta(X\cdot w^{(j)})-Y\big)$$，其中$$w^{(j)}$$为第$$j$$次迭代的结果，则第$$j+1$$次为：


$$
w^{(j+1)} \gets w^{(j)} - \eta X^T\cdot \big(\theta(X\cdot w^{(j)})-Y\big)
$$


4）转到步骤（3），一直到$$ L(w)$$满足一定条件，或者迭代到足够的次数。

在批量梯度下降算法中，每一步的迭代都需要计算所有样本，当样本数较大时，计算量会很大。

**时间复杂度：**

每次迭代更新$$X\cdot w^{(j)}$$的计算次数为$$m\times n$$，$$\theta(X\cdot w^{(j)})-Y = Z$$的计算次数为$$n$$次，$$X^T \cdot Z$$的计算次数为$$m\times n$$，则时间复杂度为$$O(m\times n)$$

假定迭代次数为$$k$$次，每次迭代更新$$w$$需要计算

#### 2.随机梯度下降（Stochastic Gradient Descent）

将上面的步骤（3）改为：

随机选取某个样本$$x^{(i)}$$，则：


$$
w^{(j+1)} \gets w^{(j)}-\eta \big(\theta(w^{(j)}\cdot x^{(i)})-y^{(i)}\big)x^{(i)}
$$


一直到迭代到足够的次数。

> 参考：
>
> [https://zh.wikipedia.org/wiki/发生比](https://zh.wikipedia.org/wiki/发生比)
>
> [http://vividfree.github.io/机器学习/2015/12/13/understanding-logistic-regression-using-odds](http://vividfree.github.io/机器学习/2015/12/13/understanding-logistic-regression-using-odds)
>
> [http://blog.csdn.net/bitcarmanlee/article/details/51473567](http://blog.csdn.net/bitcarmanlee/article/details/51473567)



