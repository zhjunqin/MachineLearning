### 方向导数

定义：若二元函数$$z=f(x,y)$$在点$$P(x_0,y_0)$$处沿着$$\vec{l}$$（方向角为$$\alpha$$，$$\beta$$）存在下列极限


$$
\dfrac{\partial f}{\partial l}=\lim\limits_{\rho\to 0}\dfrac{f(x+\Delta x, y+\Delta y)-f(x,y)}{\rho}
$$



$$
=f_x(x_0,y_0)\cos \alpha+f_y(x_0,y_0)\cos\beta
$$


其中$$\rho=\sqrt{(\Delta x)^2+(\Delta y)^2}$$，$$\Delta x=\rho\cos\alpha$$，$$\Delta y=\rho\cos\beta$$，则称$$\dfrac{\partial f}{\partial l}$$为函数在点$$P$$处沿着方向$$\vec{l}$$的方向导数。方向导数$$\dfrac{\partial f}{\partial l}$$也就是函数$$z=f(x,y)$$在点$$P$$上沿着$$\vec{l}$$的变化率。

![](/assets/方向导数2.png)

> pic source: [http://www.cnblogs.com/hithink/p/7380838.html](http://www.cnblogs.com/hithink/p/7380838.html)

### 梯度

方向导数公式$$\dfrac{\partial f}{\partial l}=\dfrac{\partial f}{\partial x}\cos \alpha+\dfrac{\partial f}{\partial y}cos\beta$$，令向量$$\vec{G}=(\dfrac{\partial f}{\partial x}, \dfrac{\partial f}{\partial y})$$，向量$$\vec{l}=(\cos\alpha, \cos\beta)$$

则$$\dfrac{\partial f}{\partial l}=\vec{G}\cdot\vec{l} =\vec{l} \cos(\vec{G}, \vec{l})$$，当$$\vec{G}$$与$$\vec{l}$$方向一致时，方向导数取最大值。

定义**向量**$$\vec{G}$$为函数$$f(P)$$在$$P$$处的**梯度**（gradient）记作$$\mathrm{grad}f(P)$$，或$$\nabla f(P)$$，即


$$
\mathrm{grad}f(P)=\nabla f(P)=(\dfrac{\partial f}{\partial x}, \dfrac{\partial f}{\partial y})
$$


其中$$\nabla=(\dfrac{\partial }{\partial x}, \dfrac{\partial }{\partial y})$$，称为向量微分算子或Nabla算子。

### 梯度的几何意义

函数在点$$P$$处的沿着梯度向量的方向导数取最大值，那么也就是该方向的变化率最大，增长速度最快。

为什么梯度的方向是函数增长最快的方向？个人理解这要从导数（或偏导数）的定义开始。

![](/assets/538px-Derivative_-_geometric_meaning.svg.png)

> pic source: 
>
> https://commons.wikimedia.org/w/index.php?curid=21999275
>
> https://zh.wikipedia.org/wiki/%E5%AF%BC%E6%95%B0

函数在某一点的可导，得到的切线的斜率为导数，而切线的导数的定义为切线的斜率，而

