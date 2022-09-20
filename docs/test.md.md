Given our data sample $x_0$ and our data distribution $p(x)$, and we gradually add a certain amount of Gaussian noise to our data, in a total of $T$ steps. That means, $x_T$ will be a pure Gaussian. We can write this in a formal way:

给定我们的数据样本$x_0$和我们的数据分布$p(x)$ ，我们逐渐向我们的数据添加一定量的高斯噪声，共分$T$步。这就是说，$x_T$将是一个纯高斯图像。这个过程可以用公式表示为：

$$
q(x_{t}|x_{t-1})=\mathcal{N}(\overbrace{x_t}^{output}, \overbrace{\sqrt{1-\beta_t}}^{mean}x_{t-1}, \overbrace{\beta_t }^{variance}\mathbb{I}), \tag{1}
$$