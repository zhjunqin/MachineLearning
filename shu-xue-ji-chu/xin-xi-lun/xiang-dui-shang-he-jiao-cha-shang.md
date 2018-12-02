### 交叉熵

在信息论中，基于相同事件测度的两个概率分布$$p$$和$$q$$的交叉熵是指，当基于一个“非自然”（相对于“真实”分布$$p$$而言）的概率分布$$q$$进行编码时，在事件集合中唯一标识一个事件所需要的平均比特数。

基于概率分布$$p$$和$$q$$的交叉熵定义为：


$$
H(p,q)=E_p[-\mathrm{log}q]
$$


对于离散分布$$p$$和$$q$$：


$$
H(p,q)=-\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}q(x)
$$


或：


$$
H(p,q)=\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}\frac{1}{q(x)}
$$


特别地，当随机变量只取两个值时，$$P(X=1)=p$$，$$P(X=0)=1-p$$，$$0\leqslant p \leqslant 1$$，则


$$
H(p,q)=-\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}q(x)
$$



$$
= -[P_p(x=1)\mathrm{log}P_q(x=1) + P_p(x=0)\mathrm{log}P_q(x=0)]
$$



$$
= -[p\mathrm{log}q + (1-p)\mathrm{log}q]
$$


### 相对熵

相对熵（relative entropy）又称KL散度（Kullback-Leibler divergence），KL距离，是两个随机分布间距离的度量，记为$$D_{KL}(p||q)$$。它度量当真实分布为$$p$$时，假设分布$$q$$的无效性。


$$
D_{KL}(p||q)=E_p[\mathrm{log}\frac{p(x)}{q(x)}]=\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}\frac{p(x)}{q(x)}
$$



$$
=\displaystyle\sum_{x\in \mathcal{X}}[p(x)\mathrm{log}p(x)-p(x)\mathrm{log}q(x)]
$$



$$
=\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}p(x)-\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}q(x)
$$



$$
= -H(p)--\displaystyle\sum_{x\in \mathcal{X}}p(x)\mathrm{log}q(x)
$$



$$
=-H(p)-E_p[\mathrm{log}q(x)]
$$



$$
=H_p(q)-H(p)
$$


其中$$H_p(q)$$即是交叉熵。

当$$p=q$$时，两者之间的相对熵$$D_{KL}(p||q)=0$$。

因此$$D_{KL}(p||q)$$的含义就是：真实分布为$$p$$的前提下，使用$$q$$分布进行编码相对于使用真实分布$$p$$进行编码所多出来的比特数。

