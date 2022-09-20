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
q(x_t|x_{t-1}))=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} +\sqrt{1-\alpha_{t-1}}\epsilon_{t-1})+\sqrt{1-\alpha_t}\epsilon_t\\
=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \overbrace{\sqrt{\alpha_t -\alpha_t\alpha_{t-1}}\epsilon_{t-1}+\sqrt{1-\alpha_t}\epsilon_t}^{merge}\\
=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}
\end{aligned}
$$










### Diffusion Models: Pros and Cons

## Credit

I have been inspired by the following blogs and thank these bloggers for their hard work.

- [Lil'Log -- What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Ayan Das -- An introduction to Diffusion Probabilistic Models](https://ayandas.me/blog-tut/2021/12/04/diffusion-prob-models.html)


## Reference

[^1]: Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International Conference on Machine Learning. PMLR, 2015: 2256-2265.

[^2]: Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in Neural Information Processing Systems, 2019, 32.
[^3]: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.