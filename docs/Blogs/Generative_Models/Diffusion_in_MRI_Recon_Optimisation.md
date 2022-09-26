# Diffusion Models in MRI Reconstruction: Algorithms, Optimisations, and Beyond -- Second Part: Optimisations

This blog is largely borrowed from papers and online tutorials. The content of the article has been compiled, understood and derived by me personally. If there are any problems please feel free to correct them. Thank you in advance!

本博客主要是受论文和网上博客启发所写的内容。文章的内容是由我个人整理、理解和推导的。如果有任何问题，请随时指正，谢谢。

In the previous post we discussed some of the inherent flaws of the Diffusion Model and its advantages and disadvantages compared to other generative models. For example, the Diffusion Model requires a large number of sampling steps and long time to generate results; and, due to its Markovian nature and tractability, the inverse process requires the same number of steps to solve as the forward process, and so on. Current research has addressed several aspects of its deficiencies, such as sampling enhancement, likelihood optimisation, optimisation of the generated results, etc.

在之前的文章中我们讨论了 Diffusion Model 内在的一些缺陷和与其他生成模型相比的优势和不足。例如，Diffusion Model 需要大量的采样步骤和时间来生成结果；并且，由于其马尔可夫的特性，逆向过程需要采用和前向过程相同的步数才可以求解，诸如此类。目前的研究已经针对多个方面对其不足提出了改进方案，例如采样增强，似然优化，生成结果优化等等 [^1] [^2]。

![](../../_media/Diffusion_Optimization_GM.png 'Figure. 1, An intuitive visualization of current popular generative models. Origin: Cao et al., 2022.')

## Problem Formulation Recap

Cao et al. [^2] proposed a unified framework for current diffusion models, both discrete (the prior state $x_T$) and continuous (the prior state $x_1$) from the starting state $x_0$, with intermediate states $x_t$. They defined the forward process as $F$ and the reverse process as $R$. Thus, the forward process can be presented by following forward transition kernels:

Cao 等人[^2]提出了一个统一的当代扩散模型框架，包括从起始状态$x_0$开始的离散（先验状态$x_T$）和连续（先验状态$x_1$），中间状态$x_t$。他们将正向过程定义为$F$，反向过程定义为$R$。因此，正向过程可以通过以下正向过渡核来呈现:

$$
\begin{aligned}
F(x,\sigma) &= F_T(x_{T-1},\sigma_T) \cdots \circ F_t(x_{t-1},\sigma_t)\cdots \circ F_1(x_0,\sigma_1) \\
x_t &=F_t(x_{t-1},\sigma_t) & \text{;For discrete case} \\
F(x,\sigma)&=F_{s1}(x_s, \sigma_{s1})\circ F_{ts}(x_t,\sigma_{ts})\circ F_{0t}(x_0,\sigma_{0t}), & 0\leq t< s \leq 1\\
x_s &= F_{ts}(x_s,\sigma_{ts}) &\text{;For continuous case}
\end{aligned}
$$

The forward process is very much similar to normalising flow. However, the difference between them is the variable noise scale $\sigma$, which controls the randomness of the whole process. On the other hand, the reverse process can be defined is the same way:

前向过程与 Normalising Flow 非常相似。然而，它们之间的区别在于可变噪声$\sigma$，它控制了整个过程的随机性。另一方面，反向过程也可以用同样的方式定义：

$$
\begin{aligned}
R(x,\sigma) &= R_1(x_{1},\sigma_1) \cdots \circ R_t(x_{t},\sigma_t)\cdots \circ R_T(x_T,\sigma_T) \\
x_{t-1} &=R_t(x_{t},\sigma_t) & \text{;For discrete case} \\
R(x,\sigma)&=R_{t0}(x_t, \sigma_{t0})\circ R_{st}(x_s,\sigma_{st})\circ R_{1s}(x_T,\sigma_{1s}), & 0\leq t< s \leq 1\\
x_t &= R_{st}(x_s,\sigma_{st}) &\text{;For continuous case}
\end{aligned}
$$

In general, the continuous process take the virtue of real-time information, which makes the continuous process obtains better performance. As for the training objective, most generative models are the same, which is keeping starting distribution $x_0$ and sample distribution $\tilde{x_0}$ as close as possible. This is implemented by maximising the log-likelihood:

一般来说，连续过程具有实时信息的优点，这使得连续过程获得更好的性能。至于训练目标，大多数生成模型都是一样的，即保持起始分布$x_0$和样本分布$\tilde{x_0}$尽可能的接近。这是通过最大化对数似然来实现的:

$$
\mathbb{E}_{F(x_0,\sigma)}[-\log R(x_T,\tilde{\sigma})]
$$

### Baselines Models

We again summarise some of the forward and reverse processes of the DDPM [^3] and Score-based Model [^4] .

我们再次总结了 DDPM [^3] 和 Score-based Model [^4] 的一些正向和反向过程。

For DDPM: 

$$
\begin{aligned}
F_t(x_{t-1},\beta_t)&:=q(x_t|x_{t-1})\\&:=\mathcal{N}\left(x_t, \sqrt{1-\beta_t}x_{t-1}, \sqrt{\beta_t}\mathbb{I}\right) &\text{;Forward step}\\
F(x_0,\beta)&:=\prod^T_{t=1}q(x_t|x_{t-1}) &\text{;Forward process}\\
R_t(x_t,\Sigma_\theta)&:=p_\theta(x_{t-1}|x_t)\\&:=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)) &\text{;Reverse step}\\
R(x_T,\Sigma_\theta)&:=p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}|x_t) &\text{;Reverse process}
\end{aligned}
$$

The DDPM objective:

$$
\mathbb{E}[-\log p_\theta(x_0)] = \underbrace{\mathbb{E}_q[D_{KL}(q(x_T|x_0)\|p(x_T))}_{L_T} + \underbrace{\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_t)}_{L_0}]
$$

where $L_T$ as the forward loss, the divergence between the forward process and the random noise distribution; $L_{t-1}$ as the reverse loss, which is the sum of divergence between posterior of forwarding step and reverse step at each step; Denote $L_0$ as the decode loss. The only item can be used to train is $L_{t-1}$, the $L_{t-1}$ can be regarded as an expectation of $L_2$ loss between two noise.The detailed derivation process can be found in the [previous blog](https://wiki.rasin.cyou/#/Blogs/Generative_Models/Diffusion_in_MRI_Recon).

其中$L_T$作为前向损失，前向过程与随机噪声分布的散度； $L_{t-1}$为反向损失，即前向步后验与反向步在每一步的散度之和； 将 $L_0$ 表示为解码损失。 唯一可以用来训练的就是$L_{t-1}$，$L_{t-1}$可以看成是两个噪声之间损失$L_2$的期望值。详细推导过程见[之前的博客](https://wiki.rasin.cyou/#/Blogs/Generative_Models/Diffusion_in_MRI_Recon)。

$$
L_{simple}:=\mathbb{E}_{x_0,\epsilon}\left\| \tilde{\epsilon}-\epsilon_\theta(\sqrt{\bar{\alpha}}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon\right\|
$$

![](../../_media/Diffusion_Optimization_DDPM.png 'Figure. 2, The DDPM architecture. Origin: Cao et al., 2022.')

For Score-Based Model:

$$
\begin{aligned}
q_\sigma(\tilde{x}|x)&:=\mathcal{N}(\tilde{x}|x,\sigma^2\mathbb{I}) &\text{; Gaussian perturbation kernel}\\
x_i &= x_{i-1}+\sqrt{\sigma^2_i-\sigma^2_{i-1}}\epsilon &\text{; Transition kernel between states}\\
L&:=\frac12 \mathbb{E}\left[\left\| s_\theta(x,\sigma) - \triangledown \log q(x) \right\|^2\right] &\text{; Score matching process}
\end{aligned}
$$

Traditional score matching techniques requires massive computational cost. Implicit score matching (ISM) [^7]  and Sliced score matching (SSM) [^8]  are proposed to optimise the score matching procedure. Denoised Score matching (DSM) proposed by Song et al. [^4] transforms the original score matching into a perturbation kernel learning by perturbing a sequence of increasing noise:

传统的分数匹配技术需要大量的计算成本。 提出了隐式分数匹配（ISM）[^7]  和切片分数匹配（SSM）[^8]  来优化分数匹配过程。 Song等人提出的去噪分数匹配（DSM）[^4]  通过扰动一系列增加的噪声，将原始分数匹配转换为扰动核学习：

$$
L:=\frac12 \mathbb{E}_{q_\sigma((\tilde{x}|x)p_{data}(x))}[\| s_\theta(\tilde{x})-\triangledown_{\tilde{x}}\log q_\sigma(\tilde{x}|x)) \|^2_2]
$$

The noise distribution is defined to be $q_\sigma(\tilde{x}|x)=\mathcal{N}(\tilde{x}|x,\sigma^2\mathbb{I})$.

$$
L(\theta;\sigma):=\frac12 \mathbb{E}_{p_{data}(x)}\mathbb{E}_{\tilde{x} \sim \mathcal{N}(x,\sigma^2\mathbb{I})}\left[\|s_\theta(\tilde{x},\sigma) +\frac{\tilde{x}-x}{\sigma^2}\|^2_2\right]
$$

![](../../_media/Diffusion_Optimization_SBM.png 'Figure. 3, The Score-based Model architecture. Origin: Cao et al., 2022.')

Song et al. [^5]  proposed the Score SDE, which generalise from the discrete setting to continuous, the diffusion process can be viewed as a continuous case described by Stochastic Differential Equation.

Song 等人[^5] 还提出了Score SDE，它从离散设置泛化为连续设置，扩散过程可以看作是随机微分方程描述的连续情况。


$$
\begin{aligned}
dx&=\underbrace{f(x,t)}_{\text{drift mean}}dt+\underbrace{g(t)}_{\text{diffusion noise}}dw, &\text{; Forward SDE}\\
F_{ts}(x(t),g_{ts})&:=q(x_s|x_t) & 0<t<s\leq1\\ 
&:=\mathcal{N}(x_s|f_{st}x_t,g^2_{st}\mathbb{I}) & \text{; Forward Process}\\
dx&=[f(x,t)-g(t)^2\triangledown_x \log p_t(x)]dt+g(t)d\bar{w}, &\text{; Reverse SDE}\\
R_{st}(x(s),g_{st})&:=q(x_t|x_s,x_0) &\text{; Reverse Process}\\
&:= \mathbb{N}\left(x_t| \frac{1}{g^2_{s0}}(f_{t0}g^2_{st}x_0+f_{st}g^2_{t0}x_s),\frac{g^2_{t0}g^2_{st}}{g^2_{s0}}\mathbb{I}   \right)\\
f_{st}&=\frac{f(x,s)}{f(x,t)},\\ g_{st}&=\sqrt{g(s)^2-f_{st}^2g(t)^2}\\
L&:=\mathbb{E}\{\lambda(t)\mathbb{E}_{x_0}\mathbb{E}_{x_tx_0}[\| s_\theta(x_t,t) - \triangledown_{x_t} \log p(x_tx_0) \|^2_2]\} &\text{; Training Objective}
\end{aligned}
$$

The SDE based method can be combined with DDPM and score-based model, the transition kernel can be expressed as:

基于 SDE 的方法可以与 DDPM 和基于分数的模型相结合，转换核可以表示为：

$$
dx = -\frac{1}{2}\beta_t x dt + \frac{\beta_t}dw \tag{1}
$$

$$
dx = \sqrt{\frac{d[\sigma^2_t]}{dt}}dw \tag{2}
$$

Moreover, Song et al. [^5]  also proposed a continuous-time ODE that supports the deterministic process. Maoutsa et al. [^6]  proposed that any type of diffusion process can be derived into a special form of ODE. In the case that functions $G$ is independent of $x$, the probability flow ODE is:

此外，Song 等人 [^5]  还提出了一种支持确定性过程的连续时间 ODE，来源于 Maoutsa等人 [^6]  提出任何类型的扩散过程都可以导出为特殊形式的 ODE。 在函数 $G$ 独立于 $x$ 的情况下，概率流 ODE 为：

$$
dx = \{f(x,t)-\frac12 G(t)G(t)^T \triangledown_x \log p_t(x)\}dt
$$

The advantage of ODE to SDE is that it can be solved with larger step size as there is no randomness inherently.

ODE 到 SDE 的优势在于它可以用更大的步长来求解，因为它本质上没有随机性。

### Sampling Algorithms

At each sampling step, the samples drawn from random noise are iterated to more closely resemble the original distribution.

在每个抽样步骤中，从随机噪声中抽取的样本将不断迭代以更接近原始分布。

DDPM uses the most basic sampling method, also known as ancestral sampling, to reconstruct the image using an inverse Markovian gradient. This algorithm randomly draws a sample from the noise distribution over the course of $T$ iterations, and at each iteration calculates its denoising result to recover the reconstructed image:

DDPM 采用的是最基础的采样方法，也被称为祖先采样，用逆向的马尔可夫梯度来重构图像。这个算法在$T$次迭代过程中随机从噪声分布中抽取一个样本，在每个迭代中都计算其去噪结果以恢复重建图像。：

![](../../_media/Diffusion_Optimization_ancestral_sampling.png 'Figure. 4, The ancestral sampling algorithm. Origin: Cui et al., 2022.')

In the Score-based model, Langevin Dynamics Sampling [^9]  is used more often. The method can just take the score function $\triangledown_x \log p(x)$ and generate samples from the probability density $p(x)$, again in an iterative fashion, with a fixed step size $\epsilon > 0$:

而在 Score-based model 中，朗之万动力学 [^9] 采样用的更多。该方法可以以固定步长 $\epsilon >0$ 只需要score function $\triangledown_x \log p(x)$ 而从概率密度 $p(x)$中产生样本，同样也是以一个迭代的方式：

![](../../_media/Diffusion_Optimization_ALDS.png 'Figure. 5, Annealed Langevin Dynamics Sampling. Origin: Cui et al., 2022.')

Song et al. [^5] inspired by a type of ODE black-box solver in order to produce high-quality samples and trade-off accuracy for efficiency for all reversed SDE. The algorithm is consists of a predictor and a corrector. The authors combined the diffusion SDE solver as predictor, with annealed Langevin dynamics above as the corrector. Moreover, the two kinds of SDE are called Variation Preserving (VP, Eq. $(1)$) and Variation Explosion (VE, Eq. $(2)$) SDE, where for VP SDE, $f_{VP}=-\frac12 \beta(t)x,\ g_{VP}=\sqrt{\beta(t)}$, and $f_{VE}=0,\ g_{VE}=\sqrt{\frac{d[\sigma^2(t)]}{dt}}$ for VE SDE.

Song等人 [^5]  受到一种 ODE 黑盒求解器的启发，以便为所有反向 SDE 生成高质量样本和权衡精度以提高效率。 该算法由预测器和校正器组成。 作者将 diffusion SDE 求解器作为预测器，将上述 Annealed Langevin Dynamics sampling 作为校正器。 此外，作者提出两种 SDE， 称为 Variation Preserving (VP, Eq. $(1)$) 和 Variation Explosion (VE, Eq. $(2)$) SDE，其中对于 VP SDE，$f_{VP}=- \frac12 \beta(t)x,\ g_{VP}=\sqrt{\beta(t)}$, $f_{VE}=0,\ g_{VE}=\sqrt{\frac{d[\ sigma^2(t)]}{dt}}$ 用于 VE SDE。

![](../../_media/Diffusion_Optimization_PC.png 'Figure. 6, Predictor-Corrector Sampling for both VE SDE and VP SDE. Origin: Cui et al., 2022.')

## Optimisations

### Speed-up Strategies

#### Training Scheme

This section mainly focuses on the improvement of the training process, rather than the sampling process, to speed up the training process without losing accuracy.

本节主要关注训练过程的改进，而不是采样过程，以加快训练过程而不损失准确性。

##### Knowledge Distillation

Knowledge distillation refers to using a smaller student network to learn the representation information of a complex teacher network, which can be more efficient and save computational costs.

知识蒸馏是指使用更小的学生网络来学习复杂教师网络的表示信息，这样可以更高效，节省计算成本。

Salimans et al. [^10] first applied this idea to the diffusion model, gradually distilling from one sampling model to another. At each distillation step, the student model continuously learns how to generate single-step results as close as the teacher model, gradually reducing the number of sampling steps. In some applications such as ProDiff [^11]  only 2 sampling steps are required to get good results.

Salimans 等人 [^10] 首先将这个想法应用于扩散模型，从一个采样模型逐渐提炼到另一个采样模型。 在每个蒸馏步骤中，学生模型不断学习如何生成与教师模型一样接近的单步结果，逐渐减少采样步骤的数量。 在 ProDiff [^11] 等某些应用中，只需 2 个采样步骤即可获得良好的结果。

##### Diffusion Scheme Learning

The idea of these methods is if there is an impact on the speed of the model if a different diffusion mode is used. TDPM [^12]  combined GAN with diffusion model to learn implicit generative distribution from random noise. Similarly, Early Stop DDPM (ES-DDPM) [^13]  used VAE to generate prior data which learned the latent space from $x_0$. Both method borrow the other faster generative model to accelerate the sampling process, and reducing sampling steps. CCDF [^14]  shows that starting from a single forward diffusion with better initialisation significantly reduces the number of sampling steps. On the other hand, Franzese et al. [^15] want to use the network training to probe the optimal steps for a certain task, which makes the training more flexibile.

本节的思想是，如果使用不同的扩散模式，是否会对模型的速度产生影响。 TDPM [^12] 将 GAN 与扩散模型相结合，从随机噪声中学习隐式生成分布；类似地，Early Stop DDPM (ES-DDPM) [^13] 使用 VAE 生成从 $x_0$ 学习潜在空间的先验数据。 这两种方法都借用了另一种更快的生成模型来加速采样过程，并减少采样步骤。 CCDF [^14] 表明，从具有更好初始化的单个前向扩散开始，显着减少了采样步骤的数量。 另一方面，Franzese 等人 [^15] 想利用网络训练来探寻某个任务的最优步骤，这使得训练更加灵活。

##### Noise Scale Designing

DDPM sets the noise level to a linear scale which can be better, while more work has been done on noisy scheduler. We can think of each sampling step as a random walk in the direction of a fixed distribution towards the target, so adjusting the noise can benefit the sampling process.

DDPM 将噪声设置为线性比例，实际上并不够好，其他工作在其噪声调度程序上做了更多文章。我们可以把每个采样步骤都看作为在固定指向目标分布方向上的一个随机游走，因此调节噪声可以有利于采样过程。

Improved DDPM [^16] adds a weight $\lambda$ in $L_{simple}$ and $L_{vlb}$ to set the noise scale. San Roman et al. [^17] added an additional noise prediction network $P_\theta$ to predict one-step noise before sampling, and the task of optimising the loss function was transformed to learn a better noise scheduler . Likewise, FastDPM [^18] and VDM [^19] express the parameters in the loss function as related to the noise scale. Different from this, Cold Diffusion [^20] proposed a new sampling method to eliminate the wrong design of the generator $\mathcal{R}$ in all samplers. The improved sampling method is:

Improved DDPM [^16] 在$L_{simple}$和$L_{vlb}$中增加了一个权重$\lambda$来设置噪声比例。San Roman et al. [^17] 额外增加了一个噪声预测网络 $P_\theta$来在采样之前对单步噪声进行预测，而优化损失函数的任务则是转变为了学习一个更好的噪声调度器。同样的，FastDPM [^18] 和 VDM [^19] 将损失函数中的参数表示为与噪声比例关联。与此不同的是，Cold Diffusion [^20] 提出了新的采样方法来消除所有采样器中对生成器$\mathcal{R}$的错误设计，其改进的采样方法为：

![](../../_media/Diffusion_Optimization_Cold_Diffusion.png 'Figure. 7, Cold diffusion sampling. Origin: Cui et al., 2022.')

#### Sampling Scheme

While the above works are effective in improving training policies and noise schedulers, retraining existing models is a slow and complicated process. It is more intuitive to change the sampling process directly on the pre-trained model, and the idea is to get better generation results with fewer steps.

虽然以上工作在改进训练策略和噪声调度器方面行之有效，但对于现有模型重新训练是一个缓慢而复杂的过程。在预训练模型上直接更改采样过程更加直观，该思想以更少的采用步数来得到更好的生成结果。

##### Implicit Sampler

In general, we need to use a reverse process that is synchronised with the forward process to get results that produce the original data. Inspired by the generative implicit model [^21], Song et al. [^22] propose DDIM, which uses an implicit and jump-step sampling method, which does not require retraining since the forward process is already determined. DDIM uses neural ODE to achieve jump-step acceleration:

一般来说，我们需要使用与前向过程相同步数的反响过程才能得到生成符合原始数据的结果。从 generative implicit model [^21]  中得到启发，Song 等人 [^22] 提出了DDIM，该方法使用隐式和跳步的采样方法，由于前向过程已经确定，这样做并不需要重新训练。DDIM使用neural ODE来实现跳步加速：

$$
d\bar{x}(t)=\epsilon^{(t)}_\theta(\frac{\bar{x}(t)}{\sqrt{\sigma^2+1}})d\sigma(t)
$$

where $\sigma_t$ is parameterised by $\sqrt{1-\alpha}/\sqrt{\alpha}$ and $\bar{x}$ is parameterised as $x/\sqrt{\alpha}$. Based on DDIM, gDDIM [^23] generalises it to the SDE framework. Different from the former, INDM [^24] uses the implicit mechanism of Normalising Flow to convert the nonlinear diffusion process into a linear latent diffusion process.


其中 $\sigma_t$ 由 $\sqrt{1-\alpha}/\sqrt{\alpha}$ 参数化，$\bar{x}$ 由 $x/\sqrt{\alpha}$ 参数化。基于DDIM, gDDIM [^23] 将其泛化到SDE框架下。与前者不同的是，INDM [^24] 使用Normalising Flow的隐式机制将非线性的扩散过哼转换为线性的隐扩散过程。

##### Differential Equation Solver Sampling

Song et al. [^5] introduced Score SDE and Probability Flow ODE into the diffusion model, which also caused a lot of repercussions, and many works also began to focus on proposing more efficient SDE and ODE solvers, using fewer steps to get a minimal approximation. Of course, there is also a trade-off between sampling speed and sampling quality. In general, ODE solvers are simpler and better to use than SDE solvers.

Song 等人 [^5]  将 Score SDE 和 Probability Flow ODE 引入到扩散模型中，也引起了不小的反响，许多工作也开始着眼于提出更加高效的 SDE 和 ODE 求解器，使用更少的步数来得到最小化的近似。当然，在采样速度和采样质量之间也存在着一个权衡。一般来说，ODE 求解器比起SDE求解器更简单也更好应用。


## Other Diffusion Variations

### Latent Diffusion Model

## Conditional Image Generation

## Credit

I have been inspired by the following blogs and thank these bloggers for their hard work.

- 


## Reference

[^1]: Yang L, Zhang Z, Hong S. Diffusion Models: A Comprehensive Survey of Methods and Applications[J]. arXiv preprint arXiv:2209.00796, 2022.

[^2]: Cao H, Tan C, Gao Z, et al. A Survey on Generative Diffusion Model[J]. arXiv preprint arXiv:2209.02646, 2022.

[^3]: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.

[^4]: Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in Neural Information Processing Systems, 2019, 32.

[^5]: Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.

[^6]: Maoutsa D, Reich S, Opper M. Interacting particle solutions of Fokker–Planck equations through gradient–log–density estimation[J]. Entropy, 2020, 22(8): 802.

[^7]: Hyvärinen A, Dayan P. Estimation of non-normalized statistical models by score matching[J]. Journal of Machine Learning Research, 2005, 6(4).

[^8]: Song Y, Garg S, Shi J, et al. Sliced score matching: A scalable approach to density and score estimation[C]//Uncertainty in Artificial Intelligence. PMLR, 2020: 574-584.

[^9]: Bakry D, Émery M. Diffusions hypercontractives[M]//Seminaire de probabilités XIX 1983/84. Springer, Berlin, Heidelberg, 1985: 177-206.

[^10]: Salimans T, Ho J. Progressive distillation for fast sampling of diffusion models[J]. arXiv preprint arXiv:2202.00512, 2022.

[^11]: Huang R, Zhao Z, Liu H, et al. Prodiff: Progressive fast diffusion model for high-quality text-to-speech[J]. arXiv preprint arXiv:2207.06389, 2022.

[^12]: Zheng H, He P, Chen W, et al. Truncated diffusion probabilistic models[J]. arXiv preprint arXiv:2202.09671, 2022.

[^13]: Lyu Z, Xu X, Yang C, et al. Accelerating Diffusion Models via Early Stop of the Diffusion Process[J]. arXiv preprint arXiv:2205.12524, 2022.

[^14]: Chung H, Sim B, Ye J C. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12413-12422.

[^15]: Franzese G, Rossi S, Yang L, et al. How much is enough? a study on diffusion times in score-based generative models[J]. arXiv preprint arXiv:2206.05173, 2022.

[^16]: Nichol A Q, Dhariwal P. Improved denoising diffusion probabilistic models[C]//International Conference on Machine Learning. PMLR, 2021: 8162-8171.

[^17]: San-Roman R, Nachmani E, Wolf L. Noise estimation for generative diffusion models[J]. arXiv preprint arXiv:2104.02600, 2021.

[^18]: Zhang Q, Chen Y. Fast Sampling of Diffusion Models with Exponential Integrator[J]. arXiv preprint arXiv:2204.13902, 2022.

[^19]: Kingma D, Salimans T, Poole B, et al. Variational diffusion models[J]. Advances in neural information processing systems, 2021, 34: 21696-21707.

[^20]: Bansal A, Borgnia E, Chu H M, et al. Cold diffusion: Inverting arbitrary image transforms without noise[J]. arXiv preprint arXiv:2208.09392, 2022.

[^21]: Mohamed S, Lakshminarayanan B. Learning in implicit generative models[J]. arXiv preprint arXiv:1610.03483, 2016.

[^22]: Song J, Meng C, Ermon S. Denoising diffusion implicit models[J]. arXiv preprint arXiv:2010.02502, 2020.

[^23]: Zhang Q, Tao M, Chen Y. gDDIM: Generalized denoising diffusion implicit models[J]. arXiv preprint arXiv:2206.05564, 2022.

[^24]: Kim D, Na B, Kwon S J, et al. Maximum Likelihood Training of Implicit Nonlinear Diffusion Models[J]. arXiv preprint arXiv:2205.13699, 2022.