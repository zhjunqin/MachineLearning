### Logistic分布

Logistic分布的定义：

设$$X$$是连续随机变量，$$X$$服从Logistic分布是指具有下列分布函数和密度函数：


$$
F(x)=P(X\leqslant x)=    \dfrac{1}{1+e^{-(x-\mu)/\gamma}}
$$



$$
f(x)=F'(X\leqslant x)=    \dfrac{e^{-(x-\mu)/\gamma}}{\gamma(1+e^{-(x-\mu)/\gamma})^2}
$$


其中，$$\mu$$为位置参数，$$\gamma \gt0$$为形状参数。

概率分布函数如下（$$\mu$$是位置函数，改变它可以平移图形）：

![](/assets/logistic_1.png)

分布函数属于Logistic函数，是一条S形曲线（sigmoid curve）。该曲线以点$$(\mu, 	\dfrac{1}{2})$$为中心对称，即满足


$$
F(-x+\mu)-	\dfrac{1}{2} = -F(x+\mu) +	\dfrac{1}{2}
$$


概率密度函数：

![](/assets/logistic_2.png)

分布函数属于Logistic函数，是

