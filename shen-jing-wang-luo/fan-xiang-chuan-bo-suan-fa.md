### 反向传播算法

我们希望有个算法，能够让我们找到权重和偏置，以至于神经网络的输出$$y(x)$$能够拟合所有的训练输入$$x$$。为了量化我们如何实现这个目标，我们定义一个代价函数：


$$
C(w,b)=\frac{1}{2n}\displaystyle\sum_{x}||y(x)-a^L(x)||^2
$$


这里$$w$$表示所有网络中权重的集合，$$b$$是所有的偏置，$$n$$是训练输入数据的个数，$$L$$表示网络的层数，$$a^L=a^L(x)$$是表示当输入为$$x$$时的网络输出的激活值向量，求和则是在总的训练输出$$x $$上进行的。符号$$||v||$$是指向量$$v$$的模。我们把$$C$$称为**二次代价函数**；有时也别称为**均方误差**或者**MSE**。

代价函数$$C(w,b)$$是非负的，因为求和公式中的每一项都是非负的。此外，当对于所有的训练输入$$x $$，$$y(x)$$接近于输出$$a$$时，代价函数的值相当小，即$$C(w,b)\approx0$$。

反向传播算法给出了一个计算代价函数梯度的的方法：

1. 输入$$x$$：为输入层设置对应的激活值$$a^1$$
2. 前向传播：对每个$$l=2,3,...,L$$计算相应的的$$z^l=w^l\cdot a^{(l-1)} + b^l$$和$$a^l = \sigma(z^l)$$
3. 输出层误差$$\delta^L$$：计算向量$$\delta^L=\nabla_aC \bigodot\sigma'(z^L)$$
4. 反向误差传播：对每个$$l= L-1, L-2, ..., 2$$，计算$$\delta^l=(({w^{(l+1)}}^T)\delta^{(l+1)})\bigodot\sigma'(z^L)$$
5. 输出：代价函数的梯度由$$\frac{\partial C}{\partial W^l_{jk}}=a^{l-1}_k\delta^l_j$$和$$\frac{\partial C}{\partial b^l_{j}}=\delta^l_j$$得出。

### 反向传播算法的证明

#### 两个假设

反向传播算法的目标是计算代价函数$$C$$分别关于$$w$$和$$b$$的偏导数$$\frac{\partial C}{\partial W^l_{jk}}$$和$$\frac{\partial C}{\partial b^l_{j}}$$。为了让方向传播可行，我们需要做出关于代价函数的两个主要假设。

第一个假设就是代价函数可以被写成一个在每个训练样本$$x$$上的代价函数$$C_x$$的均值$$C=\frac{1}{n}\displaystyle\sum_{x}C_x$$。对于二次代价函数，每个独立的训练样本的代价是$$C_x=\frac{1}{2}||y(x)-a^L(x)||^2$$，这个假设对于其他的代价函数也必须满足。需要这个假设的原因是反向传播实际上是对一个独立的训练样本计算了$$\frac{\partial C_x}{\partial w}$$和$$\frac{\partial C_x}{\partial b}$$，然后通过在所有的训练样本上进行平均化获得$$\frac{\partial C}{\partial w}$$和$$\frac{\partial C}{\partial b}$$。

第二个假设就是代价可以写成神经网络输出的函数

![](/assets/network-cost-function.png)

如图所示，将代价函数$$C$$看成仅有输出激活值$$a^L$$的函数。

#### Hadamard乘积

Hadamard 乘积是按元素乘法的运算
$$
\begin{bmatrix}
   1 \\
   2 \\
\end{bmatrix} \odot \begin{bmatrix}
   3 \\
   4 \\
\end{bmatrix}=\begin{bmatrix}
   1*3 \\
   2*4 \\
\end{bmatrix} =\begin{bmatrix}
   3 \\
   8 \\
\end{bmatrix}
$$
假设$$s$$和$$t$$是两个相同维度的向量，那么我们使用$$s \odot t $$来表示按元素的乘积。所以$$s \odot t $$的元素就是$$(s \odot t )_j=s_jt_j$$。

#### 反向传播的四个基本方程

反向传播其实是对权重和偏置变化影响代价函数过程的理解。最终的含义就是计算偏导数$$\frac{\partial C}{\partial W^l_{jk}}$$和$$\frac{\partial C}{\partial b^l_{j}}$$。为了计算这些值，我们先引入一个中间量$$\delta^l_j$$，这个称之为在$$l^{th}$$的第$$j^{th}$$个神经元上的误差。反向传播将给出计算误差$$\delta^l_j$$的流程，然后将其关联到计算上面两个偏导数上面。

假定在$$l^{th}$$层的第$$j^{th}$$神经元上，对神经元的带权输入增加很小的变化$$\Delta z^l_j$$，这使得神经元的输出由$$\sigma(z^l_j)$$变成$$\sigma(z^l_j+\Delta z^l_j)$$，这个变化会向网络后的层进行传播，最终导致整个代价产生$$\frac{\partial C}{\partial z^l_j} \Delta z^l_j$$的改变。 假如$$\frac{\partial C}{\partial z^l_j}$$是一个很大的值（或正或负），那么可以通过选择与其相反的符号的$$\Delta z^l_j$$来降低代价。相反如果$$\frac{\partial C}{\partial z^l_j}$$是一个接近于$$0$$的值，这时候并不能通过调整输入$$z^l_j$$来改善多少代价。所以这里有个启发式的认识，$$\frac{\partial C}{\partial z^l_j}$$是神经元的误差度量。

按照上面的描述，我们定义$$l^{th}$$层的第$$j^{th}$$个神经元的上的误差$$\delta^l_j$$为
$$
\delta^l_j=\frac{\partial C}{\partial z^l_j}
$$
我们使用$$\delta^l$$表示关联于$$l$$层的误差向量。

接下来我们介绍四个基本方程。











