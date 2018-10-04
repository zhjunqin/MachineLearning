### 神经网络中符号的定义

我们首先给出网络中权重、偏置和激活值的符号定义。

我们用 $$l$$ 来表示神经网络的层数，第一次层是输入的值，最后一层是输出的值。

**权重**：$$w_{jk}^l$$表示的是从$$(l-1)^{th}$$层的第$$k^{th}$$个神经元到$$l^{th}$$层的第$$j^{th}$$个神经元的链接上的权重。

**偏置**：$$b_j^l$$表示在$$l^{th}$$层的第$$j^{th}$$个神经元的偏置。

**激活值的输入**：$$z^l_j$$表示在第$$l^{th}$$的第$$j^{th}$$神经元激活值的输入。也就是
$$
z^l_j=\displaystyle\sum_{k}w^l_{jk} a^{l-1}_k + b^l_j
$$
**激活值**：$$a_j^l$$表示在$$l^{th}$$层的第$$j^{th}$$个神经元的激活值。也就是
$$
a^l_j=\sigma(z^l_j)=\sigma(\displaystyle\sum_{k}w^l_{jk} a^{l-1}_k + b^l_j)
$$




![](/assets/network-definition.png)

上面的图中 $$a_3^2 = \sigma(w_{31}^2 \cdot x_1 + w_{32}^2 \cdot x_2 + w_{33}^2 \cdot x_3 + b_3^2)$$。 也就是


$$
a_j^l = \sigma(\displaystyle\sum_{k} w_{jk}^l \cdot a_k^{l-1}+ b_j^l)=\sigma(w_{j}^l \cdot a^{l-1} + b_j^l)
$$


其求和是在第$$(l-1)^{th}$$上的所有$$k$$个神经元上进行，其中$$w_{j}^l$$表示的是第$$(l-1)^{th}$$层到第$$l^{th}$$层的第$$j$$个神经元的权重向量。

我们先将每一层用矩阵表达，假定第$$(l-1)^{th}$$有$$m$$个神经元，第$$l^{th}$$层有$$n$$个神经元，那么从第$$(l-1)^{th}$$到第$$l^{th}$$层的权重个数有$$m \times n$$个，偏置个数有$$n$$个。则：

第$$(l-1)^{th}$$的神经元向量为$$a^{l-1}= \begin{bmatrix}
   a_1^{l-1} \\
   a_2^{l-1} \\
   a_3^{l-1} \\
     ... \\
   a_m^{l-1} 
\end{bmatrix}$$，第$$l^{th}$$层的神经元向量为$$a^{l}= \begin{bmatrix}
   a_1^{l} \\
   a_2^{l} \\
   a_3^{l} \\
     ... \\
   a_n^{l} 
\end{bmatrix}$$，

偏置向量为$$b^{l}= \begin{bmatrix}
   b_1^{l} \\
   b_2^{l} \\
   b_3^{l} \\
     ... \\
   b_n^{l} 
\end{bmatrix}$$，权重矩阵为：$$W^l= \begin{bmatrix}
   w_{11} & w_{12} & w_{13} & ... & w_{1m} \\
   w_{21} & w_{22} & w_{23} & ... & w_{1m} \\
   w_{31} & w_{12} & w_{13} & ... & w_{1m} \\
                                 ... \\
   w_{n1} & w_{n2} & w_{n3} & ... & w_{nm} 
\end{bmatrix} = \begin{bmatrix}
   w_1^{l} \\
   w_2^{l} \\
   w_3^{l} \\
     ... \\
   w_n^{l} 
\end{bmatrix}$$

也就是：


$$
a^{l}= \begin{bmatrix}
   a_1^{l} \\
   a_2^{l} \\
   a_3^{l} \\
     ... \\
   a_n^{l} 
\end{bmatrix}= \begin{bmatrix}
   \sigma(w_1^l \cdot  a^{l-1} + b_1^l) \\
   \sigma(w_2^l \cdot  a^{l-1} + b_2^l) \\
   \sigma(w_3^l \cdot  a^{l-1} + b_3^l) \\
     ... \\
  \sigma(w_n^l \cdot  a^{l-1} + b_n^l)
\end{bmatrix}
$$


最后可以得到：


$$
a^l = \sigma(W^l \cdot a^{l-1} + b^l )
$$


这个表达式给出了一个更加全局的思考每层的激活值和前一层激活值的关联方式，我们仅仅用权重矩阵作用在激活值上，然后加上一个偏置向量，最后作用$$\sigma$$函数。

