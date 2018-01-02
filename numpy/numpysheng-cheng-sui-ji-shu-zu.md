> Source : http://www.cnblogs.com/plwang1990/p/3842073.html

* 生成长度为10，在\[0,1\)之间平均分布的随机数组：

```
rarray=numpy.random.random(size=10)
or
rarray=numpy.random.random((10,))
```

* 生成在-0.1到0.1之间的平均分布：

```
rarray=0.2*numpy.random.random(size=10)-0.1
or
rarray=numpy.random.uniform(-0.1,0.1,size=10)
```

* 生成长度为10，符合正态分布的随机数

```
mu,sigma=0,0.1 #均值与标准差
rarray=numpy.random.normal(mu,sigma,10)
```





