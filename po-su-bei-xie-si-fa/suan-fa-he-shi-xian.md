#### 朴素贝叶斯算法

给定数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x\in \mathcal{X}\subseteq R^n$$，$$y\in \mathcal{Y}=\{c_1, c_2,...,c_K\}$$，$$X$$是定义在输入空间$$\mathcal{X}$$上的随机向量，$$Y$$是定义在输出空间$$\mathcal{Y}$$上的随机变量，则对于输入$$x$$，计算如下的到输出$$ y$$。


$$
y=f(x)=\arg \max_{c_k}\prod_{j=1}^n P(X_j=x_j|Y=c_k)P(Y=c_k)
$$


式中的$$P(\cdot)$$值小于1，多个小于1的值连乘在python中执行会导致下溢，因此可以取对数，可以将乘法改为加法。而且对数函数是递增函数并不影响结果。则：


$$
y=f(x)=\arg \max_{c_k}ln\big(\prod_{j=1}^n P(X_j=x_j|Y=c_k)P(Y=c_k)\big)
$$



$$
=\arg \max_{c_k}\big(lnP(Y=c_k)+\displaystyle\sum_{i=1}^nln(P(X_j=x_j|Y=c_k)\big)
$$


#### 算法实现

这里考虑

> 数据源：https://github.com/apachecn/MachineLearning/tree/python-2.7/input/4.NaiveBayes/email

asd

