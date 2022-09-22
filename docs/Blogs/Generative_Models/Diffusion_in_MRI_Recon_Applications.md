# Diffusion Models in MRI Reconstruction: Algorithms, Optimisations, and Beyond -- Part Three: Applications

This blog is largely borrowed from papers and online tutorials. The content of the article has been compiled, understood and derived by me personally. If there are any problems please feel free to correct them. Thank you in advance!

本博客主要是受论文和网上博客启发所写的内容。文章的内容是由我个人整理、理解和推导的。如果有任何问题，请随时指正，谢谢。

## Diffusion Model in MRI Reconstructions

MRI reconstruction is an inverse problem, i.e. the complete MR image $x$ is obtained from the sampled measurement $y$ through the model. Several applications have successfully applied the Diffusion Model to this problem, generating accurate and reliable MR image results based on $y$ as a condition or using other methods. In the following we will take some approaches to highlight.

MRI 重构是一个逆向问题，也就是说，需要模型从采样的measurement $y$ 中得到完整的 MR 图像 $x$。一些应用已经成功地将 Diffusion Model 应用到这个问题上，根据$y$作为生成条件或者其他方法来得到准确和可靠的 MR 图像结果。以下我们将采取一些方法重点介绍。

### Score-Based Generative Models

#### Solving inverse problems in medical imaging with score-based generative models [^1]

We have learned from numerous papers that Score-based Models have good unconditional generation capabilities [^2][^3], but since it is an inverse problem, we need to generate the image domain data from the target measurement domain.

我们从众多论文中了解到，基于分数的模型具有良好的无条件生成能力[^2][^3]，但由于它是一个逆问题，我们需要从目标测量空间生成图像空间的数据。

According to this paper, let's first familiarise ourselves with the definitions. First we need to train a score-based model $s_\theta(x_t, t)$ to generate unconditional samples from a medical data prior $p(x)$. If the samples are generated conditionally, e.g. $p(x|y)$, then a conditional random process $\{x_t\}$ is obtained from the original random process $y$ based on the observation $y$. The marginal distribution for time $t$ can be expressed as $p_t(x_t, y)$, and the goal is to generate the original data $p_0(x_0|y)$, which by definition is the same as $p(x|y)$. In effect, the condition $y$ is added to the reverse process to solve a conditional reverse SDE:  

根据这篇论文，我们先熟悉一下定义。首先，我们需要训练一个score-based model $s_\theta(x_t, t)$，从医疗数据先验$p(x)$生成无条件的样本。如果样本是有条件生成的，例如$p(x|y)$，那么根据观测值$y$，从原始随机过程$y$中得到一个条件随机过程$\{x_t\}$。时间$t$的边际分布可以表示为$p_t(x_t, y)$，目标是生成原始数据$p_0(x_0|y)$，根据定义它与$p(x|y)$相同。实际上，条件$y$被添加到反向过程中，以解决一个条件反向SDE：

$$
dx_t= [f(t)x_t - g^2(t)\triangledown_{x_t} \log p_t(x_t|y)]dt + g(t)d\bar{w_t},\ t\in[0, 1]
$$

We need to calculate the score function in this equation, which of course is not trivial. One option is to re-model $s_\theta(x_t, t, y) \approx \triangledown_{x_t} \log p_t(x_t|y)$, depending explicitly on $y$. The authors argue that this would not only require paired supervised training data, but would also have the disadvantage of supervised learning [^3]. The authors argued that it is possible to leave the original unconditional score-based model unchanged, without any measurement information other than the original prior training dataset $p(x)$. Instead, the conditional information $y$ is given during the inference process, and a stochastic process $\{y_t\}$ is constructed in which noise is then gradually added to $y$. The two stochastic processes are linked using a proximal optimisation step, and the original model is used to generate intermediate samples that match the conditional information $y$.  

我们需要计算这个方程中的分数函数，这当然不是微不足道的。一个选择是重新建模$s_\theta(x_t, t, y) \approx\triangledown_{x_t} \log p_t(x_t|y)$，明确地取决于$y$。作者认为，这不仅需要成对的监督训练数据，而且也有监督学习的缺点[^3]。作者认为，可以不改变原来的无条件score-based model，除了原来的先验训练数据集$p(x)$之外，没有任何 measurement 信息。相反，在推理过程中给出了条件信息$y$，并构建了一个随机过程$\{y_t\}$，然后将噪声逐渐加入到$y$中。这两个随机过程通过近似优化步骤联系起来，原始模型被用来生成与条件信息$y$相匹配的中间样本。




#### Score-based diffusion models for accelerated MRI [^2]

#### 

### DDPM-Based Diffusion Models

#### Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction


#### 

### 


## Credit

I have been inspired by the following blogs and thank these bloggers for their hard work.


## Reference

[^1]: Song Y, Shen L, Xing L, et al. Solving inverse problems in medical imaging with score-based generative models[J]. arXiv preprint arXiv:2111.08005, 2021.

[^2]: Song Y, Ermon S. Improved techniques for training score-based generative models[J]. Advances in neural information processing systems, 2020, 33: 12438-12448.

[^3]: Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.


Chung H, Ye J C. Score-based diffusion models for accelerated MRI[J]. Medical Image Analysis, 2022: 102479.