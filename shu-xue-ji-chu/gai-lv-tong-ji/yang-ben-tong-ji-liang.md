### 统计量

设$$X_1$$，$$X_2$$，...，$$X_n$$是来自总体$$X$$（随机变量）的一个样本，它们相互独立，$$g(X_1,X_2,...,X_n)$$是$$X_1$$，$$X_2$$，...，$$X_n$$的函数，若$$g$$中不含未知参数，则称$$g(X_1,X_2,...,X_n)$$是一**统计量**。

因为$$X_1$$，$$X_2$$，...，$$X_n$$都是随机变量，而统计量是随机变量的函数，因此统计量是一个随机变量。设$$x_1,x_2,...,x_n$$是相应于样本$$X_1$$，$$X_2$$，...，$$X_n$$的样本值，则称$$g(x_1,x_2,...,x_3)$$是$$g(X_1,X_2,...,X_n)$$的**观察值**。

**样本均值**：


$$
\overline{X}=\frac{1}{n}\displaystyle\sum_{i=1}^{n} X_i
$$


**样本方差（无偏估计）**：


$$
S^2=\frac{1}{n-1}\displaystyle\sum_{i=1}^{n} (X_i-\overline{X})^2=\frac{1}{n-1}(\displaystyle\sum_{i=1}^{n} X_i^2-n\overline{X}^2)
$$


**样本标准差**：


$$
S=\sqrt{S^2}=\sqrt{\frac{1}{n-1}\displaystyle\sum_{i=1}^{n} (X_i-\overline{X})^2}
$$


**样本**$$k$$**阶（原点）距**：


$$
A_k=\frac{1}{n}\displaystyle\sum_{i=1}^{n} X_i^k
$$


**样本**$$k$$**阶中心距**：


$$
A_k=\frac{1}{n}\displaystyle\sum_{i=1}^{n} (X_i-\overline{X})^k
$$


**样本的协方差：**


$$
Cov(X,Y)=\frac{1}{n-1}\displaystyle\sum_{i=1}^{n} (X_i-\overline{X})(Y_i-\overline{Y})
$$


其中$$X_1$$，$$X_2$$，...，$$X_n$$是来自总体$$X$$的一个样本，$$Y_1$$，$$Y_2$$，...，$$Y_n$$是来自总体$$Y$$的一个样本。

**样本协方差矩阵：**

假定$$X_1$$，$$X_2$$，...，$$X_n$$是多维随机变量


$$
c_{ij}=Cov(X_{i},X_{j})=\frac{1}{n-1}\displaystyle\sum_{k=1}^{n} (X_{ik}-\overline{X_i})(X_{jk}-\overline{X_j})
$$

$$
C=\begin{bmatrix}
   c_{11} & c_{12} & ... & c_{1n}  \\
   c_{21} & c_{22} & ... & c_{2n} \\
   \vdots & \vdots & & \vdots \\
   c_{n1} & c_{n2} & ... & c_{nn} 
\end{bmatrix}
$$


> 为什么样本方差是除以$$n-1$$，而不是$$n$$？
>
> “均值已经用了$$n$$个数的平均来做估计，在求方差时，只有$$n-1$$个数和均值信息是不相关的。而第$$n$$个数已经可以由前$$n-1$$个数和均值来唯一确定，实际上没有信息量，所以在计算方差时，只除以$$n-1$$“
>
> （详细请参考 [https://www.zhihu.com/question/20099757）](https://www.zhihu.com/question/20099757）)



