# Diffusion Models in MRI Reconstruction: Algorithms, Optimisations, and Beyond

This blog is largely borrowed from papers and online tutorials. The content of the article has been compiled, understood and derived by me personally. If there are any problems please feel free to correct them. Thank you in advance!

本博客主要是借用论文和网上教程的内容。文章的内容是由我个人整理、理解和推导的。如果有任何问题，请随时指正，谢谢。

## Introduction

The idea of the diffusion model is inspired by [non-equilibrium thermodynamics](https://arxiv.org/abs/1503.03585) [^1]. Simply speaking, it first adds noise to the data and transform into a pure Gaussian distribution, then learns a noise-to-data process in discrete steps, just like in a flow-based manner. And then, Noise-Conditioned Score Network [(NCSN)](https://arxiv.org/abs/1907.05600) [^2] Denoising Diffusion Probabilistic Model [(DDPM)](https://arxiv.org/abs/2006.11239) [^3], both  have taken it forward and applied it well to the field of computer vision.

扩散模型的想法是受[非平衡热力学](https://arxiv.org/abs/1503.03585)的启发。简单地说，它首先在数据中加入噪声并转化为纯高斯分布，然后以离散的步骤学习噪声到数据的过程，就像基于流的方式。而后，Noise-Conditioned Score Network[（NCSN）](https://arxiv.org/abs/1907.05600) Denoising Diffusion Probabilistic Model[（DDPM）](https://arxiv.org/abs/2006.11239)，都将其发扬光大，很好地应用于计算机视觉领域。

Nowadays, the diffusion model has been strongly developed in several fields, such as computer vision (Image Generation,Segmentation, Image-to-Image Translation, Super Resolution, Image Editing, Text-to-Image, Medical Imaging, Video Generation, and Point Cloud), audio (Audio Generation, Audio Conversion, Audio Enhancement, and Text-to-Speech), and others (Adversarial Attack and Defense, Natural Language, Time-Series, Molecule Generation, and so on).  

如今，扩散模型在多个领域得到了大力发展，如计算机视觉（图像生成、分割、图像到图像的翻译、超分辨率、图像编辑、文本到图像、医学成像、视频生成和点云）、音频（音频生成、音频转换、音频增强和文本到语音）以及其他（对抗性攻击和防御、自然语言、时间序列、分子生成等等）。 

This blog focuses on the application of the diffusion model as a generative model for MRI reconstruction in medical images, starting with the algorithm of the diffusion model, introducing and deriving it from a shallow to a deeper level. In the process of analysing the advantages and disadvantages of the Diffusion Model some optimisation methods are introduced, and finally introducing some SOTA variants and discussing some unresolved issues and future directions.  

本博客主要讨论扩散模型作为生成模型在医学影像中MRI重构的应用。从扩散模型的算法开始，我们将由浅入深地介绍和推导，在分析Diffusion Model的优劣过程中介绍一些优化方法，最后介绍一些SOTA和相关应用，并讨论一些未解决的问题和未来的方向。 

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

#### Loss Function

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
&\leq \underbrace{D_{KL}(q(x_t|x_0)||p(x_T))}_{L_T} + \underbrace{\sum^T_{t=2}D_{KL}(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))}_{L_{t-1}}-\underbrace{\log p_\theta(x_0|x_1)}_{L_0}
\end{aligned}
$$

Take a closer look at the first item, we can see that the numerator is the forward process, and the denominator is actually a pure Gaussian distribution, which makes the first term a very small number -- which means they are very similar, and we can omit that: $L_T$is constant and can be ignored during training because $q$ has no learnable parameters and $x_T$ is a Gaussian noise;  As for the second term, both the numerator and the denominator are in the same form, a single forward and reverse process. The last term, DDPM models $L_0$ using a separate discrete decoder derived from $\mathcal{N}(x0;\mu_\theta(x_1,1),\Sigma_\theta(x1,1))$. A detailed introduction is in [Outlier's video](https://youtu.be/HoKDTa5jHvg?t=1456) and [AssemblyAI](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/).

仔细看看第一项，我们可以看到分子是正向过程，而支配者实际上是一个纯高斯分布，这使得第一项成为一个非常小的数字--这意味着它们非常相似，我们可以省略:$L_T$是常数，在训练中可以忽略，因为$q$没有可学习的参数，$x_T$是高斯噪声；至于第二项，分子和分母的形式都一样，是一个单步正向和反向过程。最后一项，DDPM 使用由$\mathcal{N}(x0;\mu_\theta(x_1,1),\Sigma_\theta(x1,1))$派生的单独的离散解码器对$L_0$建模。详细介绍见[Outlier的视频](https://youtu.be/HoKDTa5jHvg?t=1456)和[AssemblyAI](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)。

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
&=\exp \left( -\frac12 \left( \frac{x_t^2-2\sqrt{\alpha_t}x_t\color{blue}{x_{t-1}}+\color{black}{\alpha_t}\color{red}{x^2_{t-1}}}{\beta_t} +\frac{\color{red}{x^2_{t-1}\color{black}-2\sqrt{\bar{\alpha}_{t-1}}x_0\color{blue}{x_{t-1}}\color{black}+\bar{\alpha}_{t-1}x^2_0}}{1-\bar{\alpha}_{t-1}}-\frac{(x_t-\sqrt{\bar{\alpha}_tx_0})^2}{1-\bar{\alpha}_t}       \right)     \right)\\
&=\exp\left(-\frac12\left( \color{red}\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right)x^2_{t-1}  \color{black}-\color{blue}\left(\frac{2\sqrt{\alpha_t}}{\beta_t}x_t+\frac{2\sqrt{\alpha_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)x_{t-1}  \color{black}+ C(x_t, x_0)\right)\right)
\end{aligned}
$$

As in the above equation, we only need the terms associated with $x_{t-1}$ and the part associated with $C(x_t,x_0)$ is omitted. (why is this possible?) We can use the standard Gaussian distribution to get the mean and variance corresponding to that in Eq. $(9)$ as follows.:

如上式，我们只需要$x_{t-1}$相关的项，$C(x_t,x_0)$相关的部分被省略（为什么可以？）。我们可以用标准高斯分布来得到式$(9)$中该分布对应的均值和方差：

$$
\begin{aligned}
\tilde{\beta}_t&=1/\color{red}\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right)\\
&=\frac{(1-\bar{\alpha}_t)\beta_t}{\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t} &\text{;since } \beta_t=1-\alpha_t\\
&=\color{green}\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t &\text{;since } \bar{\alpha}_t=\alpha_t\cdot\bar{\alpha}_{t-1}
\\
\tilde{\mu}(x_t, x_0) &= \color{blue}\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\alpha_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right) \color{black} / \color{red}\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right)\\
&=\color{blue}\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\alpha_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)\color{green}\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\\
&=\frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1-\bar{\alpha_t}}x_t + \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\bar{\alpha}_{t}}x_0
\end{aligned}
$$

This form of the mean looks complicated, doesn't it? Once again, we can use the reparameterization trick by transforming equation $(4)$ and bringing in:

均值的这种形式看起来很复杂，对不对？我们又可以再一次利用reparameterization trick，将式4进行变换后带入：

$$
x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)
$$

$$
\begin{aligned}
\tilde{\mu}(x_t, x_0) &= \frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1-\bar{\alpha_t}}x_t +\frac{\sqrt{\bar{\alpha}}_{t-1}\beta_t}{1-\bar{\alpha}_{t}}\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)\\
&=\frac{\sqrt{\bar{\alpha}_t}\sqrt{\alpha}_t(1-\bar{\alpha}_{t-1})+\sqrt{\bar{\alpha}}_{t-1}\beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}}_t}x_t-\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{\sqrt{1-\bar{\alpha}_t}\sqrt{\bar{\alpha}_t}}\epsilon_t &\text{;we deal with the first item first}\\
&=\frac{\cancel{\sqrt{\bar{\alpha}_t}}\sqrt{\alpha}_t(1-\bar{\alpha}_{t-1})+\cancel{\sqrt{\bar{\alpha}}_{t-1}}(1-\alpha_t)}{(1-\bar{\alpha}_t)\cancel{\sqrt{\bar{\alpha}}_t}}x_t-\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{\sqrt{1-\bar{\alpha}_t}\sqrt{\bar{\alpha}_t}}\epsilon_t &\text{;Divide both by } \sqrt{\bar{\alpha}}_{t-1}\\
&=\frac{\sqrt{{\alpha}_t}\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})+(1-\alpha_t)}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}x_t-\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{\sqrt{1-\bar{\alpha}_t}\sqrt{\bar{\alpha}_t}}\epsilon_t\\
&=\frac{\cancel{\alpha_t-\bar{\alpha}_t+1-\alpha_t}}{\cancel{(1-\bar{\alpha}_t)}\sqrt{\alpha_t}}x_t -\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{\sqrt{1-\bar{\alpha}_t}\sqrt{\bar{\alpha}_t}}\epsilon_t &\text{;the first item is clear :)} \\
&=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{\cancel{\sqrt{\bar{\alpha}_{t-1}}}(1-\alpha_t)}{\sqrt{1-\bar{\alpha}_t}\cancel{\sqrt{\bar{\alpha}_t}}}\epsilon_t &\text{;work on second term the same way}\\
&=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{(1-\alpha_t)}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_t\\
&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t\right) &\text{very pretty outcome :)}
\end{aligned}
$$

Thus, we can finally conclude the $\tilde{\mu}_t$ and $\tilde{\beta}_t$ as:

因此，我们最后可以得出$\tilde{\mu}_t$和$\tilde{\beta}_t$如下：

$$
\tilde{\mu}(x_t, x_0)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t\right) \tag{11}
$$

and 

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \tag{12}
$$

Now look back to our loss function in Eq. $(10)$, this is to compare two forward and reverse process in Gaussian. We can train the neural network $\mu_\theta$ to predict $\tilde{\mu}$ in Eq. $(11)$. Since the closed form of forward process is available, we can easily get any time step data $x_t$, and then we only need to predict the noise $\epsilon_t$ from it at step $t$:

现在回头看看我们在公式$(10)$中的损失函数，这是为了比较高斯的两个正向和反向过程。我们可以训练神经网络$mu_\theta$来预测公式$(11)$中的$tilde{\mu}$。由于正向过程的封闭形式是可用的，我们可以很容易地得到任何时间步长的数据$x_t$，然后我们只需要从它的步长$t$预测噪声$epsilon_t$。

$$
\mu_\theta(x_t, t)= \frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)
$$

$$
\begin{aligned}
x_{t-1} &= \mathcal{N}(\overbrace{x_{t-1}}^{output}, \overbrace{\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)}^{mean}, \overbrace{\Sigma_\theta(x_t, t)}^{variance})\\
&=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t\epsilon_\theta(x_t,t)}}+\sqrt{1-\alpha_t}\epsilon) &\text{;Reparameterization trick}
\end{aligned}
$$

Now recalling our loss function $L_t$, we can easily see that this is the distance that minimises two different means $\tilde{\mu}$ and $\mu_\theta$:

现在回顾一下我们的损失函数$L_t$，我们就可以很容易的发现这是最小化两个不同的均值$\tilde{\mu}$和$\mu_\theta$的距离:

$$
\begin{aligned}
L_t&=\mathbb{E}\left[\frac{1}{2\|\Sigma_\theta(x_t,t)\|^2_2}\|\tilde{\mu}(x_t,t), \mu_\theta(x_t,t)\|^2\right]\\
&=\mathbb{E}\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\alpha_t)\|\Sigma_\theta\|^2_2}\|\epsilon_t-\epsilon_\theta(x_t,t)\|^2\right]
\end{aligned}
$$

They found out during their experiments that omit the scaling term brings DDPM better sampling quality and easier implementation. Thus we can have a simple form of the loss:

他们的实验中发现，省略缩放项会给DDPM带来更好的采样质量，并且更容易实现。因此，我们可以有一个简单的损失函数：

$$
\begin{aligned}
L^{simple}_t&=\mathbb{E}_{t\sim(1,T), x_0, \epsilon}[\|\epsilon_t-\epsilon_\theta(x_t,t)\|^2]\\
&=\mathbb{E}_{t\sim(1,T), x_0, \epsilon}[\|\epsilon_t-\epsilon_\theta(\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}\epsilon_0, t)\|^2]
\end{aligned}
$$

Remember we omit tons of irrelevant terms that not related to $x_{t-1}$? If you want you can add them now, because that constant $C$ is not relevant with $\theta$.

还记得我们省略了大量与$x_{t-1}$无关的项吗？如果你想的话，你现在就可以把它们加进去，因为那个常数项$C$和$\theta$没有关系。

$$
L_{simple}=L^{simiple}_t+C = =\mathbb{E}_{t\sim(1,T), x_0, \epsilon}[\|\epsilon_t-\epsilon_\theta(\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}\epsilon_0, t)\|^2] +C \tag{13}
$$

![DDPM Algorithm](../../_media/Diffusion_in_MRI_Recon_DDPM_Algo.png 'Figure. 3, The Training and Sampling algorithm from DDPM, by Ho et al.')

#### Another side of the coin: Score-Based Model

What we did above is introducing denoising diffusion probabilistic model (DDPM), which consists of two parameterised Markov chains and uses variational inference (VI) to produce samples matching the original data after a finite time. However, you also heard about something called Score-Based Model (SBM), acting in the same way like diffusion model, which makes you confuse. In fact, it is just two sides of a coin, revealing two equivalent formulations of the same phenomena.

我们上面介绍的是 Denoising Diffusion Probabilistic Model（DDPM），它由两个参数化的马尔科夫链组成，并使用 Variational Inference（VI）在有限时间后产生与原始数据相匹配的样本。然而，你也听说过一种叫做 Score-based Model（SBM）的东西，其作用方式与扩散模型相同，这使你感到困惑。事实上，这只是一枚硬币的两面，揭示了同一现象的两种等价的表述。

Song et al. [^4]'s work was detailed in his best paper at [ICLR 2021](https://www.youtube.com/watch?v=L9ZegT87QK8). It has the same setting as DDPM, the original data distribution $p_0(x)$ is perturbed by noise to the perturbed distribution $p_t(x)$, and finally get to be the prior distribution $p_T(x)$ (pure noise). Song et al. treat this process as a Stochastic Differential Equation (SDE), which is a generalisation to Ordinary Differential Equation (ODE). The process can be interpreted as gradually perturb the data distribution $p_0(x)$ to the prior distribution  $p_T(x)$ with the stochastic process for time interval $t$. The solution to the SDE can be expressed as:

Song 等人在 [ICLR 2021](https://www.youtube.com/watch?v=L9ZegT87QK8) 会议中的最佳论文中详细介绍了他的工作。它的设置与DDPM相同，原始数据分布$p_0(x)$被噪声扰动为扰动分布$p_t(x)$，最后得到的是先验分布$p_T(x)$（纯噪声）。Song等人将这一过程视为随机微分方程（SDE），它是对普通微分方程（ODE）的一种泛化。这个过程可以解释为用时间间隔$t$的随机过程逐渐扰动数据分布$p_0(x)$到先验分布$p_T(x)$。SDE的解可以表示为：

$$
dx = \underbrace{f(x,t)}_{deterministic\ drift}dt + \underbrace{g(t)}_{stochastic\ di
f\!fusion}dw, \tag{14}
$$

where $f(x,t)$ is what you would normally have in an ODE, it governs the deterministic property of the stochastic process; and the right hand term $g(t)dw$ controls stochastic diffusion, $dw$ is a brownian motion, or you can understand as a infinitesimal gaussian noise. This is a forward process, just like DDPM does. If we can learn the reverse process, we will be able to start from a random noise vector simulate the reverse stochastic process and obtain a sample from the data distribution. Perturbing data with SDE has a unique merit that is for any stochastic process given by an SDE, there exists a corresponding Reverse SDE (Anderson, 1982) that gives the reverse process, which can be written as:

其中$f(x,t)$是你在ODE中通常会有的项，它控制着随机过程的确定性属性；而右项$g(t)dw$控制随机扩散，$dw$是布朗运动，或者你可以理解为一个无限小的高斯噪声。这是一个正向过程，就像DDPM一样。如果我们能学会反向过程，我们就能从一个随机噪声向量开始模拟反向随机过程，并从数据分布中获得一个样本。用SDE来扰动数据有一个独特的优点，就是对于任何由SDE给出的随机过程，都存在一个相应的给出了反向过程的Reverse SDE（Anderson，1982），可以写成：

$$
d(x)=[f(x,t)-g^2(t) \color{red}\triangledown_x \log p_t(x)\color{black}]dt+g(t)d\bar w \tag{15}
$$

where $dt$ and $dw$ here are both negative infinitesimal time step and backward random motion. The red part $\triangledown_x \log p_t(x)$ is known as the score function of the distribution $p(t)$. Since the reverse process has a closed form, preserving data with SDE is particularly nice. In order to describe the reverse process, we need to estimate score function from data.

其中$dt$和$dw$在这里是负的无限小时间步长和后向随机运动。红色部分$\triangledown_x\log p_t(x)$被称为分布$p(t)$的 score function。由于反向过程有一个解析解，用SDE保存数据特征是特别好的。为了描述反向过程，我们需要从数据中估计score function。

You may ask why score function can use to estimate the trajectory of the process. Then we need to mention Langevin Dynamics, which is a physical concept to the mathematical modeling of the dynamics of molecular systems [^5]. Combined with the stochastic gradient descent, stochastic gradient Langevin dynamics ([Welling & Teh 2011](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.226.363)) [^6] can produce samples from a probability density $p(x)$ using only the gradients $\triangledown_x \log⁡ p(x)$ in a Markov chain of updates:

你可能会问，为什么 score function 可以用来估计过程的轨迹。那么我们就需要提到朗之万动力学，它是一个物理概念，用于分子系统动力学的数学建模[^5]。结合随机梯度下降法，随机梯度朗之万动力学（[Welling & Teh 2011](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.226.363)）[^6] 可以从概率密度$p(x)$中产生样本，只使用马尔科夫更新链中的梯度$\triangledown_x \log p(x)$：

$$
x_t = x_{t-1} + \frac{\delta}{2}\triangledown_x\log q(x_{t-1}) + \sqrt{\delta}\epsilon_t,\ \epsilon_t \sim \mathcal{N}(0, \mathbb{I})
$$

where $\delta$ is the step size. We can see this is a forward updating process. Compared to standard SGD, stochastic gradient Langevin dynamics injects Gaussian noise into the parameter updates to avoid collapses into local minima. When $T$ approaching infinity, and $\delta$ close to $0$, $x_t$ will be the true probability density $p(x)$.

其中$\delta$是步骤大小。我们可以看到这是一个向前更新的过程。与标准SGD相比，随机梯度朗之万动力学将高斯噪声注入参数更新中，以避免陷入到局部最小值。当$T$接近无穷大，且$\delta$接近$0$时，$x_t$便是真实的概率密度$p(x)$。

Given an independent identical distribution sample from marginal distribution $p_t(x)$, this sample can be obtained easily with the forward process. Then, we proposed to learn a time-dependent score-based model: $s_\theta(x, t)\approx \triangledown_x\log p_t(x)$, using deep neural network to predict the real score function. The objective function is defined as follow:

给定一个来自边际分布$p_t(x)$的独立同分布样本，这个样本可以通过正向过程轻松获得。然后，我们提出学习一个基于时间的分数模型：$s_\theta(x, t)\approx\triangledown_x\log p_t(x)$，使用深度神经网络来逼近真实的分数函数。目标函数定义如下。

$$
\min_\theta \mathbb{E}_{t\sim \mathcal{U}(0,T)}\mathbb{E}_{x\sim{p_t(x)}}\left[\lambda(t)\|\underbrace{s_\theta(x,t)}_{model\ score} - \underbrace{\triangledown_x\log p_t(x)}_{data\ score} \|^2_2\right] \tag{16}
$$

where $t$ is drawn from a uniform distribution as time step, $\lambda(t)$ is a positive weighting function, and the objective is essentially the mean Euclidean distance between the neural network prediction and the ground truth score function. However, we cannot get the ground truth of the score function. So we require some techniques such as Score Matching (Hyvarinen, 2005), which can transform the objective to an equivalent form that does not involve the ground truth of score function. Other techniques like denoising score matching ([Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)) [^7]or sliced score matching ([Song et al., 2019](https://arxiv.org/abs/1905.07088)) [^8] will not be introduced in this article due to the limit of length.

其中，$t$从均匀分布中抽取作为时间步长，$\lambda(t)$是一个正的加权函数，目标函数基本上是神经网络预测和真实 score function 之间的平均欧氏距离。然而，我们无法得到真实的 score function。因此，我们需要一些技术，如Score Matching（Hyvarinen，2005），它可以将目标转化为不涉及真实 score function 的等效形式。由于篇幅所限，本文将不介绍其他技术，如 denoising score matching（[Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)）[^7] 或 sliced score matching（[Song et al., 2019](https://arxiv.org/abs/1905.07088)）[^8]。

You may find that the objective of SDE is very similar to the Eq. $(13)$. In fact, according to the survey paper by Yang et al. [^9], if setting $s_\theta(x_t, t)=-\frac{\epsilon_\theta(x_t, t)}{2\sigma_t}$, DDPM and score based model are equal! In practice, one needs to solve the reverse SDE using discretisation, and the resulting process resembles the reverse chain in DDPM.

你可能会发现，SDE的目标与公式$(13)$非常相似。事实上，根据Yang等人的综述论文[^9]，如果设定$s_\theta(x_t, t)=-\frac{\epsilon_\theta(x_t, t)}{2\sigma_t}$，DDPM 和 score based model 是相等的! 在实践中，我们需要用离散化的方法来解决 Reverse SDE，所得到的过程类似于DDPM中的反向过程。

### Diffusion Models: Pros and Cons

Generative models have a long history of development, from the early days of VAE, to the more recent GAN, Normalising flow, Autoregressive Model and Energy-based Model, the merits of which have been debated by scholars. The emergence of Diffusion Model has increased the number of comparisons between the various generative models. Yang et al. [^9] and Croitoru et al. [^10] compared these models with diffusion model, including theirs commons and differences.

生成模型有很长的发展历史，从早期的VAE，到最近的GAN、归一化流、自回归模型和基于能量的模型，这些模型的优劣一直被学者们争论。扩散模型的出现，增加了各种生成模型之间的比较。Yang 等人 [^9] 和Croitoru 等人  [^10]将这些模型与扩散模型进行了比较，包括其共性和差异。

![Generative Overview](../../_media/Diffusion_in_MRI_Recon_Generative_Overview.png 'Overview of different types of generative models. Image from LilLog')

#### Variational AutoEncoder (VAE)

As we mentioned above, the Diffusion Model is similar to the VAE in many ways. VAE uses a pair of *encoder* and *decoder* to first encode the input information $x$ into the latent vector $z$, which is later converted back to the original space by the decoder. This is done to allow the encoder and decoder to learn the mapping between raw space and latent space. Diffusion model, in the same way, using the data is mapped to a Gaussian distribution and the reverse process learns to transform the latent representations into data. Both objective function can be derived as a lower-bound of the data likelihood, just like we did in Eq. $(7)$. Here is a good illustration that straightforwardly shows the connection between VAE and Diffusion model.

![]()


The introductory article by Luo [^11] and [post](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html) by Turner does a good job of explaining how to derive from the basic VAE to the Diffusion Model, if you are interested.




#### GAN

#### Normalising Flow

#### Autoregressive Model

#### Energy-Based Model





-   **Pros**: Tractability and flexibility are two conflicting objectives in generative modeling. Tractable models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. Flexible models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. Diffusion models are both analytically tractable and flexible
    
-   **Cons**: Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite expensive in terms of time and compute. New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.

### MRI Reconstruction: A Quick Tutorial

## Optimisations



## Diffusion Model in MRI Reconstructions

### 


## Credit

I have been inspired by the following blogs and thank these bloggers for their hard work.

- [Lil'Log -- What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Ayan Das -- An introduction to Diffusion Probabilistic Models](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)
- [Maciej Domagała -- The recent rise of diffusion-based models](https://maciejdomagala.github.io/generative_models/2022/06/06/The-recent-rise-of-diffusion-based-models.html)
- [Ryan O'Connor -- Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
- [Angus Turner -- Diffusion Models as a kind of VAE](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html)


## Reference

[^1]: Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International Conference on Machine Learning. PMLR, 2015: 2256-2265.

[^2]: Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in Neural Information Processing Systems, 2019, 32.

[^3]: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.
[^4]: Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.
[^5]: Wikipedia, Langevin Dynamics, https://en.wikipedia.org/wiki/Langevin_dynamics
[^6]: Welling M, Teh Y W. Bayesian learning via stochastic gradient Langevin dynamics[C]//Proceedings of the 28th international conference on machine learning (ICML-11). 2011: 681-688.
[^7]: Vincent P. A connection between score matching and denoising autoencoders[J]. Neural computation, 2011, 23(7): 1661-1674.
[^8]: Song Y, Garg S, Shi J, et al. Sliced score matching: A scalable approach to density and score estimation[C]//Uncertainty in Artificial Intelligence. PMLR, 2020: 574-584.
[^9]: Yang L, Zhang Z, Hong S. Diffusion Models: A Comprehensive Survey of Methods and Applications[J]. arXiv preprint arXiv:2209.00796, 2022.
[^10]: Croitoru F A, Hondru V, Ionescu R T, et al. Diffusion Models in Vision: A Survey[J]. arXiv preprint arXiv:2209.04747, 2022.
[^11]: Luo C. Understanding Diffusion Models: A Unified Perspective[J]. arXiv preprint arXiv:2208.11970, 2022.