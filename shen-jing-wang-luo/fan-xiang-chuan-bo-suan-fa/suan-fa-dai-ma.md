### 反向传播算法代码

给定一个大小为$$m$$的小批量数据，在小批量数据的基础上应用梯度下降学习算法：

1. **输入训练样本的集合**
2. **对于每个训练样本**$$x$$：设置对应的输入激活$$a^{x,1}$$，并执行下面的步骤  
   1. **前向传播**：对于每一个$$l=2,3,...,L$$，计算$$z^{x,l}=w^l a^{x,l-1} + b^l$$和$$a^{x,l}=\sigma(z^{x,l})$$

   1. **输出误差**$$\delta^{x,L}$$：计算向量$$\delta^{x,L}=\nabla_a C_x \odot \sigma'(z^{x,L})$$

   2. **反向传播误差**：对于每一个$$l=L-1,L-2,...,2$$计算$$\delta^{x,l}=((w^{l+1})^T \delta^{x, l+1}) \odot \sigma'(z^{x,l})$$

3. **梯度下降**：对于每一个$$l=L-1, L-2, ..., 2$$根据$$w^l\to w^l - \frac{\eta}{m}\displaystyle\sum_{x}\delta^{x,l} (a^{x,l-1})^T$$和$$b^l\to b^l - \frac{\eta}{m}\displaystyle\sum_{x}\delta^{x,l} $$更新权重和偏置。



### Python 代码

aa

