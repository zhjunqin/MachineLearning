### 朴素贝叶斯法的参数估计

朴素贝叶斯法需要估计$$P(Y=c_k)$$和$$P(X_j=x_j|Y=c_k)$$


$$
y=f(x)=\arg \max_{c_k}\prod_{j=1}^n P(X_j=x_j|Y=c_k)P(Y=c_k)
$$


假定数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x\in \mathcal{X}\subseteq R^n$$，$$y\in \mathcal{Y}=\{c_1, c_2,...,c_K\}$$。

#### 极大似然估计

应用极大似然估计法估计响应的概率。

**先验概率**$$P(Y=c_k)$$的极大似然估计是：


$$
P(Y=c_k)=\dfrac{\displaystyle\sum_{i=1}^mI(y_i=c_k)}{m},\ \ \  k=1,2,...,K
$$


其中$$I(y_i=c_k)$$是指示函数，当$$y_i=c_k$$时值为1，其他情况下为0。$$m$$为数据集里的数据量。

假定输入的$$n$$维特征向量$$x$$的第$$j$$维可能的取值为$$\{x_{j1},x_{j2},...x_{js_{j}}\}$$，则**条件概率**$$P(X_j=x_{jl}|Y=c_k)$$的极大似然估计是：


$$
P(X_j=x_{jl}|Y=c_k)=\dfrac{\displaystyle\sum_{i=1}^mI(x_j^{(i)}=x_{jl},y_i=c_k)}{\displaystyle\sum_{i=1}^mI(y_i=c_k)}
$$


