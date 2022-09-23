# Diffusion Models in MRI Reconstruction: Algorithms, Optimisations, and Beyond -- Part Three: Applications

This blog is largely borrowed from papers and online tutorials. The content of the article has been compiled, understood and derived by me personally. If there are any problems please feel free to correct them. Thank you in advance!

本博客主要是受论文和网上博客启发所写的内容。文章的内容是由我个人整理、理解和推导的。如果有任何问题，请随时指正，谢谢。

## Diffusion Model in MRI Reconstructions

MRI reconstruction is an inverse problem, i.e. the complete MR image $x$ is obtained from the sampled measurement $y$ through the model. Several applications have successfully applied the Diffusion Model to this problem, generating accurate and reliable MR image results based on $y$ as a condition or using other methods. In the following we will take some approaches to highlight.

MRI 重构是一个逆向问题，也就是说，需要模型从采样的measurement $y$ 中得到完整的 MR 图像 $x$。一些应用已经成功地将 Diffusion Model 应用到这个问题上，根据 $y$ 作为生成条件或者其他方法来得到准确和可靠的 MR 图像结果。以下我们将采取一些方法重点介绍。

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

Song et al. represent the MRI reconstruction in a slightly different way: The linear operator $A$ has a full rank, i.e. $rank(A)=\min\{n,m\}=m$, then there exists a invertible matrix $T \in \mathbb{R}^{n\times n}$, a diagonal matrix $\Lambda \in \{0, 1\}^{n\times n}$ with $rank(\Lambda)=m$, and an dimensionality reduction operator $\mathcal{P}(\Lambda)\in\{0, 1\}^{m\times n}$ which removes each $i$-th element if $\Lambda_{ii}=0$. In more general terms, $A$ can be represented as a transform from the image domain to the measurement domain that contains a Fourier Transform $T$, a mask matrix $\Lambda$ and finally a $\mathcal{P}$ operator to compress the resulting downsampled data into a tighter matrix $y$.
Song等人以一种略微不同的方式表示MRI重建。线性算子$A$具有全等级，即 $rank(A)=min\{n,m\}=m$，那么存在一个可逆矩阵$T\in \mathbb{R}^{n\times n}$，一个对角矩阵$\Lambda\in \{0, 1\}^{n\times n}$，$rank(\Lambda)=m$。和一个降维算子$\mathcal{P}(\Lambda)\in\{0, 1\}^{m\times n}$，如果$\Lambda_{ii}=0$，则删除每个$i$元素。更通俗的来说，$A$可以表示为一个从图像域到测量域的一个变换，这个变换包含了傅立叶变换$T$，一个遮罩矩阵$\Lambda$，最后还有一个$\mathcal{P}$操作将得到的降采样数据压缩成更紧致矩阵$y$。

![](../../_media/Diffusion_Application_song_MRI.png 'Fig. 1, Linear measurement processes for undersampled MRI. Origin: Song et al., 2021')

We now return to how we can use the unconditional $s_\theta(x_t, t)$ to get a conditional output based on $y$. The basic idea here is to combine the conditional $y$ into the sampling process of the unconditional score-based model. Recall that we have the forward process:

现在我们回到如何使用无条件的 $s_\theta(x_t, t)$ 来获得基于$y$的条件输出。这里的基本想法是把条件 $y$ 结合到无条件score-based model 的采样过程中。回顾一下正向过程:

$$
p_{0t}(x_t|,x_0)= \mathcal{N}(x_t|\alpha(t)x_0, \beta^2(t)\mathbb{I}) \tag{1}
$$

Given the unconditional stochastic process $\{x_t\}$, we can define $y_t$ in a simple way. Here we have:

给定无条件的随机过程$\{x_t\}$，我们可以用一种简单的方式定义$y_t$：

$$
\begin{aligned}
y_0&=Ax_0+\alpha(0)\epsilon, \\
&=Ax_0+\epsilon &\text{;The definition of inverse problem}\\
y_t&=A(x_t)+\alpha(t)\epsilon, 
\end{aligned}
$$

According to Eq. $(1)$, we can use reparameterisation trick to turn $x_t$ into：

根据公式$(1)$，我们可以使用重新参数化的技巧，将$x_t$转换为

$$
x_t=\alpha(t)x_0+\beta(t)z,\ z\sim \mathcal{N}(0,\mathbb{I}) \tag{2}
$$

Then we will use $x_t$ in Eq $(2)$ to the definition:

那么我们就用公式 $(2)$中的 $x_t$ 带入到定义：
$$
\begin{aligned}
y_t&=A(\alpha(t)x_0)+\beta(t)z)+\alpha(t)\epsilon\\
&=\alpha(t)Ax_0+\beta(t)Az+\alpha(t)\epsilon\\
&=\alpha(t)(y-\epsilon)+\beta(t)Az+\alpha(t)\epsilon &\text{;}y_0\text{ is }y\\
&=\alpha(t)y+\beta(t)Az
\end{aligned}
$$

The equation above proved that $y_t$ is only concerned with $t$ and its original $y$. Thus, we can simply draw a sample $z$ and then compute $\hat{y_t}=\alpha(t)t+\beta(t)Az$.

上面的方程证明 $y_t$ 只和 $t$ 和它的原始 $y$ 有关。因此，我们可以简单地抽取一个样本 $z$ ，然后计算$\hat{y_t}=\alpha(t)t+\beta(t)Az$。

Thus, the previous unconditional sample process is iterated according to:

$$
\hat{x}_{t_i-1}=h(\hat{x}_{t_i}, z_i,s_{\theta^*}(\hat{x}_t, t)),\ i=N, N-1,\dots, 1 \tag{3}
$$

Here the iteration function $h$ takes a noisy sample $x_{t_i}$ and reduces the noise therein to generate $x_{t_{i-1}}$ , using the unconditional score model $s_{\theta^*}(x_t,t)$. Samples obtained by this procedure $\{\hat{x}_t\}$ constitute an approximation of $\{x_t\}$, where the last sample $\hat{x}_{t_0}$ can be viewed as an approximate sample from $p_0(x)$. To enforce the constraint implied by $\{y_t|y\}$, we prepend an additional step to the iteration in Eq. $(3)$, leading to:

迭代函数 $h$ 取一个有噪声的样本 $x_{t_i}$，并减少其中的噪声以生成 $x_{t_{i-1}}$，使用无条件的 score-based model $s_{\theta^*}(x_t,t)$。通过这个过程得到的样本 $\{\hat{x}_t\}$ 构成了 $\{x_t\}$ 的近似值，其中最后一个样本 $hat{x}_{t_0}$ 可以被看作是 $p_0(x)$ 的近似样本。为了增强 $\{y_t|y\}$ 所隐含的约束，我们在公式 $(3)$ 的迭代中附加了一个步骤：

$$
\hat{x}_{t_i}'=k(\hat{x}_{t_i}, \hat{y}_{t_i}, \lambda) \tag{4}
$$
$$
\hat{x}_{t_{i-1}}=h(\hat{x}_{t_i}', z_i, s_{\theta^*}(\hat{x}_{t_i},t_i)), \tag{5}
$$

where $0\leq \lambda \leq 1$ is a hyper-parameter, meaning the data consistency, or how much can we trust the condition $y_t$. Here is a illustration for the iterative process.

其中$0\leq \lambda \leq 1$是一个超参数，代表数据一致性，或者说我们对条件$y_t$ 的信任程度。下面是一个迭代过程的图示。

![](../../_media/Diffusion_Application_song_MRI_iteration.png 'Fig. 2, The iterative process of conditional sampling. Origin: Song et al., 2021')

Eq. $(4)$ promotes data consistency by solving a proximal optimisation step that minimises the distance between $\hat{x_{t_i}}'$ and $\hat{x}_{t_i}$, and the distance between $\hat{x_{t_i}}'$ and the hyperplane $\{x\in\mathbb{R}^{n\times n}|Ax=\hat{y}_{t_i}\}$. $\lambda$ is the hyperparameter used to trade of between the two:

公式 $(4)$ 通过解决一个 proximal optimisation 步骤来促进数据的一致性，该步骤使 $\hat{x_{t_i}}'$ 与 $\hat{x}_{t_i}$ ，以及 $\hat{x_{t_i}}$ 与超平面 ${x\in\mathbb{R}^{n\times n}|Ax=\hat{y}_{t_i}}$ 之间的之间的距离最小。$\lambda$是用于在这两者之间进行平衡的超参数。

$$
\hat{x_{t_i}}' = \underset{z \in \mathbb{R}^n}{\arg\min}\left\{(1-\lambda)\|z-\hat{x_{t_1}}\|^2_T+\underset{\min}{u\in\mathbb{R}^n}\lambda \|z-u\|^2_T\right\} \tag{6}
$$

The norm $|a|^2_T:=|Ta|^2_2$ is simplified for analysis. And then the closed form of Eq. $(6)$ can be derived:

规范$|a|^2_T:=|Ta|^2_2$被简化以进行分析。然后可以得出公式$(6)$的解析解：

$$
\hat{x}_{t_i}'=T^{-1}[\underbrace{\lambda \Lambda \mathcal{P}^{-1}(\Lambda)\hat{y}_{t_i}}_{from\ condition\ y}+\underbrace{(1-\lambda)\Lambda T\hat{x}_{t_i}}_{from\ x}+\underbrace{(\mathbb{I}-\Lambda)T\hat{x}_{t_i}}_{unsampled\ part}] \tag{7}
$$

The detailed derivation process can be found in the appendix section of the original paper and will not be delved into here.

详细的推导过程可以见原论文的附录部分，这里不做深究。

![](../../_media/Diffusion_Application_song_MRI_DC.png 'Fig. 3, using data consistency in MRI reconstruction, Origin: Song et al., 2021')

The hyperparameter $\lambda$ is important for balancing between $\hat{x}_{t_i}'\approx \hat{x}_{t_i}'$ and $A\hat{x}_{t_i}'\approx \hat{y}_{t_i}'$. The authors used Bayesian optimisation to tune this $\lambda$ automatically on a validation dataset. Here is an overview of the proposed unconditional sampling and inverse problem solving.

超参数$\lambda$对于平衡$\hat{x}_{t_i}'\approx \hat{x}_{t_i}'$和$A\hat{x}_{t_i}'\approx \hat{y}_{t_i}$之间的关系非常重要。作者用贝叶斯优化法在验证数据集上自动调整这个$\lambda$。下面是对所提出的无条件抽样和逆向问题解决的算法概述。

![](../../_media/Diffusion_Application_song_MRI_Algo.png 'Fig. 4, An overview of proposed algorithms. Origin: Song et al., 2021.')

##### Experiments

###### Dataset

- The Brain Tumour Segmentation (BraTS) 2021 dataset [^4]
- Single-coil setup

###### Network and Training

- NCSN++ model architecture [^3]
- Perturb the data using Variance Exploding (VE) SDE
	- $f(x)=0, g(x)=\sqrt{\frac{d(\sigma^2(t))}{dt}},\ \sigma(t)>0$
- Use Predictor-Corrector (PC) samplers 
	- use 1000 noise scales and 1 step of Langevin dynamics per noise scale, totalling $1000+1000=2000$ steps of score model evaluation.


##### Pros and Cons

###### *Pros*:

- This method is the minimal modification to a trained score-based models using iterative sampling method.
	- Just add three lines of pseudo-code (see Fig. $4$ above.)
- This method is not limited to Annealed Langevin dynamics (ALD) and outperform previous ALD method [^5].
- This method does not need to implement a supervised paradigm with paired training data, as the result outperform previous SOTA supervised methods.
	- ![](../../_media/Diffusion_Application_song_MRI_evaluation.png 'Fig. 5, Results for undersampled MRI reconstruction on BraTS. First two methods are supervised learning techniques trained with 8  acceleration. The others are unsupervised techniques. Original: Song et al., 2021.')
- This method has good generalisation to different number of measurements. 
	- Evidence from Song et al. [^1]: " ... we achieve the best performance on undersampled MRI for both $24\times$ and $4\times$ acceleration factors, whereas *DuDoRNet* fails to generalise when the acceleration factor changes."

###### *Cons*:

- 123




#### Score-based diffusion models for accelerated MRI [^6]



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

[^4]: Bakas S, Akbari H, Sotiras A, et al. Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features[J]. Scientific data, 2017, 4(1): 1-13.

[^5]: Jalal A, Arvinte M, Daras G, et al. Robust compressed sensing mri with deep generative priors[J]. Advances in Neural Information Processing Systems, 2021, 34: 14938-14954.

[^6]: Chung H, Ye J C. Score-based diffusion models for accelerated MRI[J]. Medical Image Analysis, 2022: 102479.