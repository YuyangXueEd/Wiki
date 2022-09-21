# Diffusion Models in MRI Reconstruction: Theory, Algorithms, and Beyond

This blog is largely borrowed from papers and online tutorials. The content of the article has been compiled, understood and derived by me personally. If there are any problems please feel free to correct them. Thank you in advance!

本博客主要是借用论文和网上教程的内容。文章的内容是由我个人整理、理解和推导的。如果有任何问题，请随时指正，谢谢。

## Introduction

The idea of the diffusion model is inspired by [non-equilibrium thermodynamics](https://arxiv.org/abs/1503.03585) [^1]. Simply speaking, it first adds noise to the data and transform into a pure Gaussian distribution, then learns a noise-to-data process in discrete steps, just like in a flow-based manner. And then, Noise-Conditioned Score Network [(NCSN)](https://arxiv.org/abs/1907.05600) [^2] Denoising Diffusion Probabilistic Model [(DDPM)](https://arxiv.org/abs/2006.11239) [^3], both  have taken it forward and applied it well to the field of computer vision.

扩散模型的想法是受[非平衡热力学](https://arxiv.org/abs/1503.03585)的启发。简单地说，它首先在数据中加入噪声并转化为纯高斯分布，然后以离散的步骤学习噪声到数据的过程，就像基于流的方式。而后，Noise-Conditioned Score Network[（NCSN）](https://arxiv.org/abs/1907.05600) Denoising Diffusion Probabilistic Model[（DDPM）](https://arxiv.org/abs/2006.11239)，都将其发扬光大，很好地应用于计算机视觉领域。

Nowadays, the diffusion model has been strongly developed in several fields, such as computer vision (Image Generation,Segmentation, Image-to-Image Translation, Super Resolution, Image Editing, Text-to-Image, Medical Imaging, Video Generation, and Point Cloud), audio (Audio Generation, Audio Conversion, Audio Enhancement, and Text-to-Speech), and others (Adversarial Attack and Defense, Natural Language, Time-Series, Molecule Generation, and so on).  

如今，扩散模型在多个领域得到了大力发展，如计算机视觉（图像生成、分割、图像到图像的翻译、超分辨率、图像编辑、文本到图像、医学成像、视频生成和点云）、音频（音频生成、音频转换、音频增强和文本到语音）以及其他（对抗性攻击和防御、自然语言、时间序列、分子生成等等）。 

This blog focuses on the application of the diffusion model as a generative model for MRI reconstruction in medical images, starting with the algorithm of the diffusion model, introducing and deriving it from a shallow to a deeper level, and finally introducing some SOTA variants and discussing some unresolved issues and future directions.  

本博客主要讨论扩散模型作为生成模型在医学影像中MRI重构的应用。从扩散模型的算法开始，我们将由浅入深地介绍和推导，最后介绍一些SOTA的变体，并讨论一些未解决的问题和未来的方向。 

## Background

### Diffusion Model: What is it?

Imagine that our data is a complex distribution and it may be difficult for us to get the full information about this distribution directly and completely. But like the 'Diffusion' described in thermodynamics, it is relatively easy to diffuse from a complex, high-density region to a simple, low-density region, such as a Gaussian distribution with 0 mean and 1 variance. If we can learn the inverse of this high to low process, then we can potentially generate our complex data from a simple Gaussian distribution.

想象一下，我们的数据是一个复杂的分布，我们可能很难直接、完整地获得这个分布的全部信息。但就像热力学中描述的“扩散”一样，从一个复杂的、高密度的区域扩散到一个简单的、低密度的区域是相对容易的，比如一个均值为0、方差为1的高斯分布。如果我们能学会这个从高到低过程的逆过程，那么我们就有可能从一个简单的高斯分布产生我们的复杂数据。

#### Forward Process

Given our data sample $x_0$ and our data distribution $p(x)$, and we gradually add a certain amount of Gaussian noise to our data, in a total of $T$ steps. That means, $x_T$ will be a pure Gaussian. We can write this in a formal way:

给定我们的数据样本$x_0$和我们的数据分布$p(x)$ ，我们逐渐向我们的数据添加一定量的高斯噪声，共分$T$步。这就是说，$x_T$将是一个纯高斯图像。这个过程可以用公式表示为：

$$
q(x_{t}|x_{t-1})=\mathcal{N}(\overbrace{x_t}^{output}, \overbrace{\sqrt{1-\beta_t}}^{mean}x_{t-1}, \overbrace{\beta_t }^{variance}\mathbb{I}), \tag{1}
$$

We call this process as the **forward process**. This is a single forward step, where $x_t$ is the output of the single forward step, the Gaussian we added is in a $\sqrt{1-\beta_t}$ mean, $\beta_t$ variance form. For a simple explanation, we use a linear scheduler, where the $\beta_t$ in each step is fixed. This is normally called **Gaussian Diffusion**.

我们称这个过程为**前向过程**。这是一个单一的前向步骤，其中$x_t$是单一前向步骤的输出，我们添加的高斯是以均值为$\sqrt{1-\beta_t}$，方差为$\beta_t$的形式出现的。为了便于解释，我们使用一个线性调度器，其中每一步中的$\beta_t$是固定的。这通常被称为**高斯扩散**。

Since the forward process can be seen as a Markov chain, each step is only concerned with the previous step, we can see them as a cumulative product process. Then, the whole forward process can be interpreted as:

由于前向过程可以被看作是一个马尔可夫链，每一步都只和上一步有关，我们可以把它们看作是一个累乘的过程。那么，整个前向过程可以被写作：

$$
q(x_{1:T}|x_0) = \prod^T_{t=1}q(x_t|x_{t-1}) \tag{2}
$$

A simple illustration can be used here to represent the forward process. the smaller the $t$, the clearer the image; the larger the $t$, the closer the image is to Gaussian noise.

这里可以用一个简单的插图来表示前进过程。$t$越小，图像越清晰；$t$越大，图像越接近高斯噪声。

![Fig. 1 ](../../_media/Diffusion_in_MRI_Recon_Forward.png 'Fig. 1, The forward process, adding noise to each step.')

Usually we do not use neural network in forward process, just use noise scheduler. If we set $\beta_0=0.0001$, and $\beta_T=0.02$, in a total time steps of $T=1000$, we can easily get $\beta_t$ of every step in a linear scheduler.

通常我们在前进过程中不使用神经网络，只是使用噪声调度器。如果我们设置$\beta_0=0.0001$，$\beta_T=0.02$，在总时间步数为$T=1000$时，我们可以很容易地得到线性调度器中每一步的$\beta_t$。

![Fig. 2, Noise scheduler](../../_media/Diffusion_in_MRI_Recon_beta_mean_variance.png 'Fig. 2, Noise scheduler')

Now we can rewrite the single forward step $q(x_t|x_{t-1})$ in a more convenient way, defining $\alpha_t = 1-\beta_t,\ \  \bar{\alpha_t}=\prod^T_{s=1} \alpha_s$:

我们可以定义$\alpha_t = 1-\beta_t,\ \bar{\alpha_t}=\prod^T_{s=1} \alpha_s$，那么现在我们可以用更方便的方式重写单个前向步骤$q(x_t|x_{t-1})$：

$$
q(x_t|x_{t-1})=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t \tag{3}
$$

Here we introduce $\epsilon_t$ as a $\mathcal{N}(0,\mathbb{I})$, and with reparameterization trick, which is introduced in [Lil'Log](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick), the  we can easily expand the single step equation to the whole forward process:

这里我们把$\epsilon_t$作为$\mathcal{N}(0,\mathbb{I})$引入，用重参数化技巧，也就是在[Lil'Log](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick)中介绍的，我们可以很容易地把单步方程扩展到整个前进过程。

$$
q(x_t|x_0) = \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_{0}}x_0 + \sqrt{1-\alpha_t\alpha_{t-1}\dots\alpha_0}\epsilon_0 
\\
=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}\epsilon_0 \tag{4}
$$

The first half of this formula is easier to understand, all of $x_t$ can be replaced by the next $\sqrt{\alpha_t}x_{t-1}$, so it is a cumulative multiplication of $\sqrt{\alpha_{t,0}}$. The second term is more difficult to understand and would be complicated to expand directly, but we know that the formula for the fusion of a multivariate Gaussian distribution is $\sigma'^2=\sigma_1^2+\sigma_2^2$, so in practice the expansion would be:

这个公式的前半部分比较容易理解，所有的$x_t$都可以被下一个$\sqrt{\alpha_t}x_{t-1}$代替，所以是$\sqrt{\alpha_{t,0}}$的累乘。第二项比较难理解，如果直接展开会很复杂，但我们知道多方差高斯分布的融合公式为$\sigma’^2=\sigma_1^2+\sigma_2^2$，因此实际上展开的话会是：

$$
\begin{aligned}
q(x_t|x_{t-1})) &=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} +\sqrt{1-\alpha_{t-1}}\epsilon_{t-1})+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \overbrace{\sqrt{\alpha_t -\alpha_t\alpha_{t-1}}\epsilon_{t-1}+\sqrt{1-\alpha_t}\epsilon_t}^{merge}\\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\epsilon_{t-1}
\end{aligned}
$$

Thus we can get the closed form of equation $(4)$. Usually, we can afford a larger update step when the sample gets noisier, so $\beta_1<\beta_2<\dots<\beta_T$ and therefore $\bar{\alpha}_1>\dots>\bar{\alpha}_T$.

因此我们可以得到方程$(4)$的解析解。通常情况下，当样本变得更加嘈杂时，我们可以承受更大的更新步骤，所以$\beta_1<\beta_2<\dots<\beta_T$，因此$\bar{\alpha}_1>\dots>\bar{\alpha}_T$。

#### Reverse Process

Can’t we just define a reverse process $q(x_{t−1}∣x_t)$ and trace back from the noise to the image? 
> First of all, that would fail conceptually, as we want to have a neural network that learns how to deal with a problem - we shouldn’t provide it with a clear solution. And second of all, we cannot quite do that, as **it would require marginalisation over the entire data distribution.** To get back to the starting distribution $q(x_0)$ from the noised sample we would have to marginalise over all of the ways we could arise at $x_0$ from the noise, including all of the latent states. That means calculating $\int q(x_{0:T})dx_{1:T}$, which is intractable. -- From [[Maciej Domagała Blog](https://maciejdomagala.github.io/generative_models/2022/06/06/The-recent-rise-of-diffusion-based-models.html)]

我们就不能定义一个反向过程$q(x_{t-1}∣x_t)$并从噪声追溯到图像吗？
> 首先，这在概念上是行不通的，因为我们想让一个神经网络学习如何处理问题--我们不应该给它提供一个明确的解决方案。其次，我们不能完全做到这一点，因为**它需要对整个数据分布进行边缘化。** 为了从噪声样本中回到起始分布$q(x_0)$，我们必须对所有可能从噪声中产生$x_0$的方式进行边际化，包括所有的潜藏状态。这意味着要计算$\int q(x_{0:T})dx_{1:T}$，这是很难做到的。 --来自[[Maciej Domagała Blog](https://maciejdomagala.github.io/generative_models/2022/06/06/The-recent-rise-of-diffusion-based-models.html)]

But we can approximate it by training a learnable neural network, which can approximate the reverse process. We note the reverse process model as $p_\theta(\cdot)$, which can be defined as:

但我们可以通过训练一个可学习的神经网络来近似它，它可以近似反向过程。我们注意到反向过程模型为$p_\theta(\cdot)$，它可以被定义为:

$$
p_\theta(x_{0:T})=p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}|x_t) \tag{5}
$$

Each single step of a reverse process can be seen as a kind of Gaussian:

反向过程的每一个单一步骤都可以被看作是一个高斯分布:

$$
p_\theta (x_{t-1}|x_t) = \mathcal{N}(\overbrace{x_{t-1}}^{output}, \overbrace{\mu_\theta(x_t, t)}^{mean}, \overbrace{\Sigma_\theta(x_t, t)}^{variance}) \tag{6}
$$

According to our noise scheduler above, the $\Sigma_\theta$ here is fixed, so here we only need to predict the mean $\mu_\theta$, which can be approximated by a neural network.

根据我们上面的噪声调度器，这里的$\Sigma_\theta$是固定的，所以这里我们只需要预测平均值$\mu_\theta$，它可以由一个神经网络来近似。

## Loss Function

We can now start by looking at the loss function of this model. A detailed and dynamic derivation can be found in the video [Diffusion Models | Paper Explanation | Math Explained](https://youtu.be/HoKDTa5jHvg?t=573).We find its negative cross-entropy for the target distribution $-\log(p_\theta(x_0))$. Since it is intractable, and the diffusion model can also be treated as a  Markovian hierarchical VAE, we can use variational lower bound (VLB) to optimise the loss function:

我们现在可以先来看看这个模型的损失函数。我们对目标分布求其负交叉熵：$-\log(p_\theta(x_0))$。详细且动态的推导过程可以参考视频 [Diffusion Models | Paper Explanation | Math Explained](https://youtu.be/HoKDTa5jHvg?t=573).由于它是难以计算的，而且Diffusion Model 也可以被当作马尔科夫多层VAE，我们可以使用 Variational Lower Bound（VLB）来优化损失函数：

$$
\begin{aligned}
-\log (p_\theta(x_0)) \leq -\log(p_\theta(x_0)) + D_{KL}(q(x_{1:T}|x_0) || p_\theta(x_{1:T}|x_0))
\end{aligned} \tag{7}
$$

The $D_{KL}$ is defined as:

KL散度的定义为：

$$
D_{KL}(p||q)=\int_x p(x)\log\frac{p(x)}{q(x)}dx
$$

The KL-Divergence cannot smaller than $0$, which means the two distribution are exactly the same. So we can add the KL-divergence to estimate the lower bound. We can rewrite the KL-Divergence as:

由于KL散度不能小于$0$，这意味着两个分布完全相同。所以我们可以加上KL散度来估计VLB。我们可以重写KL散度为：

$$
\begin{aligned}
\log\left(\frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T}|x_0)}\right) &=\log\left(\frac{q(x_{1:T}|x_0)}{\underbrace{\frac{p_\theta(x_0|x_{1:T})p_\theta(x_{1:T})}{p_\theta(x_0)}}_{Bayesian\ Rule}}\right)\\
&= \log(\frac{q(x_{1:T}|x_0)}{\frac{p_\theta(x_{0,T})}{p_\theta(x_0)}})\\
&= \log(\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0,T})})+\log(p_\theta(x_0))
\end{aligned}
$$

Then we can easily cancel the first item in the equation $(7)$, the VBL now is:

然后我们可以很容易地消掉方程$(7)$中的第一项，现在，VBL为：

$$
-log(p_\theta(x_0)) \leq \log(\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0,T})}) \tag{8}
$$

We can also rewrite the right hand term in equation $(8)$ as:

我们也可以将方程$(8)$中的右边项改写为:

$$
\begin{aligned}
\log\left(\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right) &= \log \left(\frac{\prod^T_{t=1}q(x_t|x_{t-1})}{p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}|x_t)}\right)\\
&=\overbrace{-log(p(x_T))}^{can\ be\ calculate}+\log\left(\frac{\prod^T_{t=1}q(x_t|x_{t-1})}{\prod^T_{t=1}p_\theta(x_{t-1}|x_t)}\right) & \text{ ;forward and reverse in log}\\
&=-log(p(x_T))+\sum^T_{t=1}\log \left(\frac{q(x_t|x_{t-1})}{p_\theta(x_t|x_{t-1})}\right) &\text{ ;log characterristic}
\end{aligned}
$$

It looks like we're at our wits' end, but the author uses a little trick here to separate out $x_0$:

看起来我们已经束手无策了，但是作者在这里使用了一个小技巧，把$x_0$分离出来：

$$
\begin{aligned}
\sum^T_{t=1}\log \left(\frac{q(x_t|x_{t-1})}{p_\theta(x_t|x_{t-1})}\right) &= \sum^T_{t=2}\log\left(\frac{q(x_t|x_{t-1})}{p_\theta(x_t|x_{t-1})}\right) + \log\left(\frac{q(x_1|x_{0})}{p_\theta(x_1|x_{0})}\right)
\end{aligned}
$$

The reason for this is that both $q(x_{t-1}|x_{t})$ and $q(x_t)$ have very high variance, and we can add an $x_0$ as a condition for all terms except for the $t=1$ case to make the variance lower and easier to determine. Another reason is that conditioning $x_0$ on itself is meaningless and leads the equation into a weird loop.

这样做的原因是因为$q(x_{t-1}|x_{t})$和$q(x_t)$都有非常高的方差，我们可以给除了$t=1$情况下的其他项添加一个$x_0$ 作为条件，使其方差降低，更容易确定。另一个理由是给$x_0$自身作条件是没有意义的而且会使等式陷入奇怪的循环。

$$
q(x_t|x_{t-1})=\frac{q(x_{t-1}|x_t)q(x_t)}{q(x_{t-1})}=\frac{q(x_{t-1}|x_t, x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}
$$

We bring this result into the original equation $(8)$:

我们将该结果带入原式 $(8)$ 中：

$$
\begin{aligned}
-\log p_\theta(x_0) &\leq -\log p(x_T) + \sum^T_{t=2}\log \left(\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{p_\theta(x_{t-1}|x_t)q(x_{t-1}|x_0)}\right) + \log \left(\frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\right) \\
&\leq -\log p(x_T) + \sum^T_{t=2}\log \left(\frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\right) + \sum^T_{t=2}\log \left(\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}\right) +  \log \left(\frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\right)
\end{aligned}
$$

If you expand the term $\sum^T_{t=2}\log \left(\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}\right)$ here you can see that it will cancel the last term's numerator, and then we can combine the remainder to the first term, just like this:

如果你把$\sum^T_{t=2}\log \left(\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}\right)$这里展开，你可以看到它将取消最后一项的分子，然后我们可以把余下的部分合并到第一项，就像这样:

$$
\begin{aligned}
-\log p_\theta(x_0) &\leq \log \frac{q(x_t|x_0)}{p(x_T)} + \sum^T_{t=2}\log \left(\frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\right) - \log p_\theta(x_0|x_1)\\
&\leq D_{KL}(q(x_t|x_0)||p(x_T)) + \sum^T_{t=2}D_{KL}(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))-\log p_\theta(x_0|x_1)
\end{aligned}
$$

Take a closer look at the first item, we can see that the numerator is the forward process, and the denominator is actually a pure Gaussian distribution, which makes the first term a very small number -- which means they are very similar, and we can omit that;  As for the second term, both the numerator and the denominator are in the same form, a single forward and reverse process. 

仔细看看第一项，我们可以看到分子是正向过程，而支配者实际上是一个纯高斯分布，这使得第一项成为一个非常小的数字--这意味着它们非常相似，我们可以省略；至于第二项，分子和分母的形式都一样，是一个单步正向和反向过程。

It is noteworthy that the reverse conditional probability is tractable when conditioned on $x_0$, then we can omit the step $t$:

值得注意的是，当以$x_0$为条件时，反向条件概率是可求解的，我们可以省略步骤$t$:

$$
p_\theta(x_{t-1}|x_t, x_0)=\mathcal{N}(x_{t-1}, \tilde{\mu}_\theta(x_t, x_0), \tilde{\beta}\mathbb{I}) \tag{9}
$$

According to the equation $(1)$ and $(9)$, we can transform the term into a form of Gaussian.

根据方程$(1)$和$(6)$，我们可以将该项转化为高斯的形式。

$$
\sum^T_{t=2}D_{KL}(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t)) = \sum^T_{t=2} \log \left(\frac{\mathcal{N}(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t\mathbb{I})}{\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta \mathbb{I})}\right)  \tag{10}
$$

According to Bayes' Theorem, we can get the following deduction (check more from [Lil'Log -- What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)): 

根据贝叶斯定理，我们可以得到以下推论（参阅[Lil'Log -- What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)）：

Since $q(x_{t-1}|x_t)=\frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)}$, we can get:

$$
\begin{aligned}
q(x_{t-1}|x_t, x_0) &=q(x_t|x_{t-1}, x_0)\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}
\end{aligned}
$$

And in fact we can think of these steps as the process of combining and subtracting several Gaussian distributions, with proportional results:

而实际上我们可以把这几步看作是几个高斯分布合并和相减的过程，结果是成正比的：

$$
\begin{aligned}
q(x_{t-1}|x_t, x_0) &=q(x_t|x_{t-1}, x_0)\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} \\
&\propto  \exp\left(-\frac12 \left( \frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\sqrt{\bar{\alpha}_{t-1}x_0})^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t-\sqrt{\bar{\alpha}_tx_0})^2}{1-\bar{\alpha}_t}  \right)\right) \\
&=\exp \left( -\frac12 \left( \frac{x_t^2-2\sqrt{\alpha_t}x_t\color{blue}{x_{t-1}}+\color{white}{\alpha_t}\color{red}{x^2_{t-1}}}{} +\frac{}{}-\frac{}{}      \right)     \right)
\end{aligned}
$$








---














### Diffusion Models: Pros and Cons

## Credit

I have been inspired by the following blogs and thank these bloggers for their hard work.

- [Lil'Log -- What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Ayan Das -- An introduction to Diffusion Probabilistic Models](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)
- [Maciej Domagała -- The recent rise of diffusion-based models](https://maciejdomagala.github.io/generative_models/2022/06/06/The-recent-rise-of-diffusion-based-models.html)


## Reference

[^1]: Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International Conference on Machine Learning. PMLR, 2015: 2256-2265.

[^2]: Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in Neural Information Processing Systems, 2019, 32.
[^3]: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.