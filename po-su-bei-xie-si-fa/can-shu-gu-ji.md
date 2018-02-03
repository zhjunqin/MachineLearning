### 朴素贝叶斯法的参数估计

朴素贝叶斯法需要估计参数$$P(Y=c_k)$$和$$P(X_j=x_j|Y=c_k)$$


$$
y=f(x)=\arg \max_{c_k}\prod_{j=1}^n P(X_j=x_j|Y=c_k)P(Y=c_k)
$$


假定数据集$$T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$$，其中$$x\in \mathcal{X}\subseteq R^n$$，$$y\in \mathcal{Y}=\{c_1, c_2,...,c_K\}$$。$$X$$是定义在输入空间$$\mathcal{X}$$上的$$n$$维随机向量，$$Y$$是定义在输出空间$$\mathcal{Y}$$上的随机变量。

#### 极大似然估计

应用极大似然估计法估计相应参数。

**先验概率**$$P(Y=c_k)$$的极大似然估计是：


$$
P(Y=c_k)=\dfrac{\displaystyle\sum_{i=1}^mI(y^{(i)}=c_k)}{m},\ \ \  k=1,2,...,K
$$


其中$$I(y_i=c_k)$$是指示函数，当$$y_i=c_k$$时值为1，其他情况下为0。$$m$$为数据集里的数据量。

假定输入的$$n$$维特征向量$$x$$的第$$j$$维可能的取值为$$\{x_{j1},x_{j2},...x_{js_{j}}\}$$，则**条件概率**$$P(X_j=x_{jl}|Y=c_k)$$的极大似然估计是：


$$
P(X_j=x_{jl}|Y=c_k)=\dfrac{\displaystyle\sum_{i=1}^mI(x_j^{(i)}=x_{jl},y^{(i)}=c_k)}{\displaystyle\sum_{i=1}^mI(y^{(i)}=c_k)}
$$



$$
j=1,2,...,n;\  l=1,2,...,s_j;\  k=1,2,...,K
$$


其中$$x_j^{(i)}$$是第$$i$$个样本的第$$j$$个特征可能取的第$$l$$个值；$$I$$为指示函数。

这里证明一下先验概率$$P(Y=c_k)$$的极大似然估计（参考 [https://www.zhihu.com/question/33959624](https://www.zhihu.com/question/33959624）。) ）。

令参数$$P(Y=c_k)=\theta_k，\ k=1,2,...,K$$。则随机变量$$Y$$的概率可以用参数来表示为$$P(Y)=\displaystyle\sum_{k=1}^K\theta_kI(Y=c_k)$$，其中$$I$$是指示函数。极大似然函数


$$
L(\theta_k;y^{(1)},y^{(2)},...,y^{(m)})=\displaystyle\prod_{i=1}^mp(y^{(i)})=\displaystyle\prod_{k=1}^K\theta_k^{t_k}
$$


其中$$m$$是样本总数，$$t_k$$为样本中$$Y=c_k$$的样本数目，满足$$\displaystyle\sum_{k=1}^Kt_k=m$$。取对数得到


$$
ln(L(\theta_k))=\displaystyle\sum_{k=1}^Kt_kln\theta_k
$$


要求该函数的最大值，同时有约束条件$$\displaystyle\sum_{k=1}^K\theta_k=1$$。利用拉格朗日乘子法，


$$
l(\theta_k,\lambda)=\displaystyle\sum_{k=1}^Kt_kln\theta_k+\lambda(\displaystyle\sum_{k=1}^K\theta_k-1)
$$


求导可以得到
$$
\dfrac{\partial l(\theta_k,\lambda)}{\partial \theta_k}=\dfrac{t_k}{\theta_k}+\lambda=0
$$
得到：
$$
t_k=-\lambda{\theta_k},\ k=1,2,...,K
$$


将所有的$$K$$个式子加起来，得到$$\displaystyle\sum_{k=1}^Kt_k=-\displaystyle\sum_{k=1}^K\lambda{\theta_k}$$，同时结合约束条件$$\displaystyle\sum_{k=1}^K\theta_k=1$$，可得$$\lambda=-m$$。最终可得
$$
P(Y=c_k)=\theta_k=\dfrac{t_k}{m}
$$
证明完毕。其他更多的详细证明请参考以上链接。



