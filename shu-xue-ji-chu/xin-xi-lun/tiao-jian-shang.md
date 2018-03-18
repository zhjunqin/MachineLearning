条件熵

> 参考 [https://zh.wikipedia.org/wiki/条件熵](https://zh.wikipedia.org/wiki/条件熵)

假设有随机变量$$(X,Y)$$，其联合概率分布为：$$P(X=x_i, Y=y_i)=p_{ij}$$

$$P(X=x_i, Y=y_j)=p_{ij}$$，$$i=1,2,...,n; j=1,2,...,m$$

条件熵描述了在已知随机变量$$X$$的值的前提下，随机变量$$Y$$ 的信息熵还有多少。同其它的信息熵一样，条件熵也用Sh、nat、Hart等信息单位表示。基于$$X$$  条件的$$Y$$ 的信息熵，用$$H(Y|X)$$表示。

$$H(Y|X=x)$$为随机变量$$Y$$在$$X$$取特定值$$x $$下的熵，那么$$H(Y|X)$$就是$$H(Y|X=x)$$在$$X$$取遍所有可能$$x$$后取平均期望的结果。

给定随机变量$$X \in \mathcal{X}$$，$$Y\in \mathcal{Y}$$，在给定$$X$$条件下$$Y$$的条件熵定义为：


$$
H(X|Y)=\displaystyle\sum_{x\in \mathcal{X}}p(x)H(Y|X=x)
$$



$$
=-\displaystyle\sum_{x\in \mathcal{X}}p(x)\displaystyle\sum_{y\in \mathcal{Y}}p(y|x)\mathrm{log}p(y|x)
$$



$$
=-\displaystyle\sum_{x\in \mathcal{X}}\displaystyle\sum_{y\in \mathcal{Y}}p(x,y)\mathrm{log}p(y|x)
$$



$$
=-\displaystyle\sum_{x\in \mathcal{X},y\in \mathcal{Y}}p(x,y)\mathrm{log}\dfrac{p(x,y)}{p(x)}
$$



$$
=-\displaystyle\sum_{x\in \mathcal{X},y\in \mathcal{Y}}p(x,y)\mathrm{log}\dfrac{p(x,y)}{p(x)}
$$



$$
=\displaystyle\sum_{x\in \mathcal{X},y\in \mathcal{Y}}-p(x,y)\mathrm{log}p(x,y)-\displaystyle\sum_{x\in \mathcal{X}}-p(x)\mathrm{log}p(x)
$$



$$
=H(X,Y)-H(X)
$$


