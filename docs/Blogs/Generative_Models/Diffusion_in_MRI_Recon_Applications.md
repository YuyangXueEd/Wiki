# Diffusion Models in MRI Reconstruction: Algorithms, Optimisations, and Beyond -- Part Three: Applications

This blog is largely borrowed from papers and online tutorials. The content of the article has been compiled, understood and derived by me personally. If there are any problems please feel free to correct them. Thank you in advance!

本博客主要是受论文和网上博客启发所写的内容。文章的内容是由我个人整理、理解和推导的。如果有任何问题，请随时指正，谢谢。

## Diffusion Model in MRI Reconstructions

MRI reconstruction is an inverse problem, i.e. the complete MR image $x$ is obtained from the sampled measurement $y$ through the model. Several applications have successfully applied the Diffusion Model to this problem, generating accurate and reliable MR image results based on $y$ as a condition or using other methods. In the following we will take some approaches to highlight.

MRI 重构是一个逆向问题，也就是说，需要模型从采样的measurement $y$ 中得到完整的 MR 图像 $x$。一些应用已经成功地将 Diffusion Model 应用到这个问题上，根据 $y$ 作为生成条件或者其他方法来得到准确和可靠的 MR 图像结果。以下我们将采取一些方法重点介绍。

### Score-Based Generative Models

#### Solving inverse problems in medical imaging with score-based generative models [^1]

We have learned from numerous papers that Score-based Models have good unconditional generation capabilities [^2] [^3], but since it is an inverse problem, we need to generate the image domain data from the target measurement domain.

我们从众多论文中了解到，基于分数的模型具有良好的无条件生成能力 [^2] [^3]，但由于它是一个逆问题，我们需要从目标测量空间生成图像空间的数据。

According to this paper, let's first familiarise ourselves with the definitions. First we need to train a score-based model $s_\theta(x_t, t)$ to generate unconditional samples from a medical data prior $p(x)$. If the samples are generated conditionally, e.g. $p(x|y)$, then a conditional random process $\{x_t\}$ is obtained from the original random process $y$ based on the observation $y$. The marginal distribution for time $t$ can be expressed as $p_t(x_t, y)$, and the goal is to generate the original data $p_0(x_0|y)$, which by definition is the same as $p(x|y)$. In effect, the condition $y$ is added to the reverse process to solve a conditional reverse SDE:  

根据这篇论文，我们先熟悉一下定义。首先，我们需要训练一个score-based model $s_\theta(x_t, t)$，从医疗数据先验$p(x)$生成无条件的样本。如果样本是有条件生成的，例如$p(x|y)$，那么根据观测值$y$，从原始随机过程$y$中得到一个条件随机过程$\{x_t\}$。时间$t$的边际分布可以表示为$p_t(x_t, y)$，目标是生成原始数据$p_0(x_0|y)$，根据定义它与$p(x|y)$相同。实际上，条件$y$被添加到反向过程中，以解决一个条件反向SDE：

$$
dx_t= [f(t)x_t - g^2(t)\triangledown_{x_t} \log p_t(x_t|y)]dt + g(t)d\bar{w_t},\ t\in[0, 1]
$$

We need to calculate the score function in this equation, which of course is not trivial. One option is to re-model $s_\theta(x_t, t, y) \approx \triangledown_{x_t} \log p_t(x_t|y)$, depending explicitly on $y$. The authors argue that this would not only require paired supervised training data, but would also have the disadvantage of supervised learning [^3] . The authors argued that it is possible to leave the original unconditional score-based model unchanged, without any measurement information other than the original prior training dataset $p(x)$. Instead, the conditional information $y$ is given during the inference process, and a stochastic process $\{y_t\}$ is constructed in which noise is then gradually added to $y$. The two stochastic processes are linked using a proximal optimisation step, and the original model is used to generate intermediate samples that match the conditional information $y$.  

我们需要计算这个方程中的分数函数，这当然不是微不足道的。一个选择是重新建模$s_\theta(x_t, t, y) \approx\triangledown_{x_t} \log p_t(x_t|y)$，明确地取决于$y$。作者认为，这不仅需要成对的监督训练数据，而且也有监督学习的缺点[^3] 。作者认为，可以不改变原来的无条件score-based model，除了原来的先验训练数据集$p(x)$之外，没有任何 measurement 信息。相反，在推理过程中给出了条件信息$y$，并构建了一个随机过程$\{y_t\}$，然后将噪声逐渐加入到$y$中。这两个随机过程通过近似优化步骤联系起来，原始模型被用来生成与条件信息$y$相匹配的中间样本。

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

因此，之前的无条件采样过程是根据以下形式进行迭代的：


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
	- Evidence from Song et al. [^1] :" ... we achieve the best performance on undersampled MRI for both $24\times$ and $4\times$ acceleration factors, whereas *DuDoRNet* fails to generalise when the acceleration factor changes."

###### *Cons*:

- Only considered the single-coil setting, with real-valued data



#### Score-based diffusion models for accelerated MRI [^6]

Shortly after this, Chung et al. [^6]  also proposed an alternative score-based model for MRI reconstruction. This method requires training a single function with *magnitude* image only, using the denoising score matching loss, construct a solver for the reverse SDE from the VE-SDE, enables to sample from $p(x|y)$. The unconditional sampling also adapt the Predictor-Corrector algorithm from Song et al. [^3] . Additionally, the Corrector steps correct the direction of gradient ascent with corrector algorithms, such as Langevin MC [^7] . 

此后不久，Chung等人 [^6]  也提出了另一种基于分数的MRI重建模型。这种方法只需要用 *幅度* 图像训练一个单一的方程，使用 denoising score matching 损失，从 VE-SDE 中构建一个 Reverse SDE 的求解器，实现从 $p(x|y)$ 中采样。无条件取样也适应Song等人的Predictor-Corrector算法 [^3]  。此外，修正器的步骤用修正器算法修正梯度上升的方向，如Langevin MC [^7]  。

![](../../_media/Diffusion_Application_Chung_MRI_PC.png 'Fig. 6, The Predictor-Corrector sampling algorithm. Origin: Chung et al., 2022')

Similar to Song et al. [^1] , they also imposed data consistency step at every iteration, after the unconditional update step. However, there are some differences of the way of adding data consistency. Chung et al omit the noise in the inverse problem which makes $y = Ax, \ A:=\mathcal{P}_\Omega\mathcal{FS}$, where $\mathcal{S}$ is the sensitivity map for each coil, $\mathcal{F}$ is the Fourier Transform, and $\mathcal{P}_\Omega$ is a diagonal matrix represents subsampling pattern with mask $\Omega$. They use another form of data consistency in Predictor step that:

与Song等人 [^1]  提出的类似，他们也在每次迭代中，在无条件更新步骤之后，施加了数据一致性步骤。然而，增加数据一致性的方式有一些不同。Chung等人在逆问题中省略了噪声，使$y=Ax, \ A:=\mathcal{P}_\Omega\mathcal{FS}$，其中$\mathcal{S}$是每个线圈的灵敏度图，$\mathcal{F}$是傅里叶变换，$\mathcal{P}_\Omega$是一个对角矩阵，代表带有掩码$\Omega$的子取样模式。他们在Predictor步骤中使用另一种形式的数据一致性，即:

$$
x_i \leftarrow x_i + \lambda A^*(y-Ax_i)=(\mathbb{I}-\lambda A^*Ax_i)+\lambda A^*y \tag{8}
$$

where $A^*$ denotes the Hermitian adjoint of $A$. After that, the Corrector step also needs another data consistency:

其中$A^*$表示$A$的赫米特邻接矩阵。之后，Corrector 步骤也需要另一个数据一致性：

$$
x_{i+1} \leftarrow x_i + \lambda A^*(y-Ax)
$$

The whole algorithm with data consistency is described as follows:

带有数据一致性的完整算法描述如下：

![](../../_media/Diffusion_Application_Chung_MRI_PC_real.png 'Fig. 7, Score-based sampling with data consitency when λ=1. Origin: Chung et al., 2022.')

During Chung's experiments, they found out this only works well in real image domain, rather than the complex signal which the practical MR Imaging used. Since the original theory of score-based SDE did not consider complex signals, another thing is that not only handling complex input may hurt the computation efficiency, but also hinder the convenience of using only magnitude data for training. The solution they gave is to implement the algorithm separately on both magnitude and phase image, treat them both like real numbers. This sounds trivial but the results shows even better than supervised model. The algorithm is shown below:

在Chung的实验中，他们发现这只在实数图像域中效果良好，而不是在真实的MR成像中使用的复数信号。由于基于分数的SDE的原始理论没有考虑复数信号，并且，处理复数输入不仅可能损害计算效率，而且还妨碍了只使用幅度数据进行训练的便利性。他们给出的解决方案是在幅值和相位图像上分别实现该算法，把它们都当成实数。这听起来微不足道，但结果显示甚至比监督模型更好。该算法如下所示：

![](../../_media/Diffusion_Application_Chung_MRI_PC_SENSE.png 'Fig. 8, Score-based sampling with data consitency works on complex image, when λ=1. Origin: Chung et al., 2022.')

In the multi-coil images, they consider the SOSS type and the hybrid type methods, respectively. SOSS does not require a prediction of the sensitivity of each coil, but uses the sum-of-root-sum-of-squares (SOSS) directly, which is easier to implement. However, the disadvantage of this is that each coil is independent of each other and no correlation between them is taken into account. Therefore, the hybrid type does a one-step aggregation of the predicted results after $m$ runs, and does a data consistency of the integrated data. The authors found experimentally that the hybrid approach worked well on the 1D sampling pattern, while SOSS worked better on the 2D sampling pattern. However, a big problem is that if $c$ coils are computed separately, the computational effort increases linearly by $c$ times. The authors propose that if GPU resources are sufficient, all coils can be computed in parallel to reduce the computational pressure.

在多线圈图像中，它们分别考虑了SOSS形式和混合形式的方法。其中，SOSS不需要对每个线圈的灵敏度进行预测，而是直接使用它们的平方根之和的平方，这样实现起来更加便捷。但这样的坏处是每个线圈互相独立，没有考虑到它们之间的相关性。因此，混合形式在$m$个运行步骤之后对预测好的结果进行一步整合，将整合好的数据做一次数据一致性。作者经过实验发现，混合方法在一维采样模式上效果好，而SOSS在二维采样模式下效果更优。然而，一个很大的问题是，如果分别计算$c$个线圈，计算量就线性增加了$c$倍。作者提出如果GPU资源充足的情况下，可以对所有线圈并行计算来减缓计算压力。

![](../../_media/Diffusion_Application_Chung_MRI_PC_SOSS_hybrid.png 'Fig. 9, Score-based sampling with data consitency works on multi-coil data, when λ=1. Origin: Chung et al., 2022.')

##### Experiments

###### Dataset

- fastMRI knee data [^8] as training data
	- dropped the first and last five slices from each volume, to avoid training the model with noise-only data.
- fastMRI+ [^12]  for anomaly detection

###### Network and Training

- Base the implementation of the time-dependent score function model ncsnpp [^3] ,
	- stems from U-Net, and the sub-block which consist U-Net are adopted from residual blocks of BigGAN [^9] ,
	- The skip connections in the residual blocks are scaled by $1/\sqrt2$
	- For pooling and unpooling, we adopt anti-aliasing pooling of Zhang et al. [^10] ,
	- 4 different levels of scale, with 4 residual networks at each level.
	- Conditioning of network with the time index $t$ is performed with Fourier features [^11] , where the conditional features are added to the encoder features.
	- ![](Diffusion_Application_Chung_MRI_PC_architecture.png 'Fig. 10, Detailed Network structure of score-based model. Origin: Chung et al., 2022.')
- use $N = 2000; M = 1$ iterations for inference as default.

##### Pros and Cons

###### *Pros*

- Agnostic to the sub-sampling pattern used in the acceleration procedure
- Can be extended to complex-valued MR image acquisition
- Applied to practical multi-coil settings with the same function
- Generalisation capability is far greater for OOD data, different contrast, and different anatomy
	- Evidence: "we are able to achieve high fidelity reconstructions regardless of the anatomy and contrast. While other methods such as U-Net and DuDoRNet generalizes to a certain extent, we can clearly observe leftover aliasing artifacts."
	- Evidence from Jalal et al. [^5] : " ... partially proved that posterior sampling is indeed highly robust to distribution shifts. This property is indeed very advantageous in real-world settings, since one may be able to use a single neural network regardless of the specific anatomy and contrast."
- High acceleration rate deviate much from the result, which shows uncertainty
	- Evidence: "First, the sample starts from a randomly sampled vector $x_N$. Second, both predictor and corrector steps involve sampling random noise vectors and adding them to the estimate. Therefore, the iterative procedure of the proposed algorithms typically converges to different outcomes."
- Generally beats SOTA methods
	- ![](../../_media/Diffusion_Application_Chung_MRI_results.png 'Fig. 10, PSNR results of x4 and x8 acceleration rate reconstruction, CCDF outperforms the other methods. Origin: Chung et al., 2022.')

###### *Cons*

- Slow training speed:
	- Evidence: "Optimization was performed for 100 epochs, and it took about 3 weeks of training the score function with a single RTX 3090 GPU."
- Slow inference speed:
	- Evidence: "Summing up, this results in about 10 minutes of reconstruction time for real-valued images, and 20 minutes of reconstruction time for complex-valued images."
	- Can speed up sampling speed by using CCDF proposed by Chung et al. [^13]. 
		- one can start to apply reverse diffusion from a forward-diffused image from better an initialization to achieve reconstruction performance that is one par or better.
- Reconstruction artifacts
	- Evidence: "... when we attempt to reconstruct OOD data with 1D under-sampling pattern, we sometimes observe mild aliasing like artifacts in local edges ... such an artifact is not observed in 2D sampling patterns ... "
- High acceleration rate leading to problems
	- Evidence: " ... when pursuing extreme-condition reconstruction ... we occasionally acquire results that are unsatisfactory (e.g. sample marked with the red dotted line). Moreover, we observe that the detailed structure has high variance within the posterior samples, due to the high illposedness. Hence, care should be taken when pushing the accelerating factor to very high values, by, for example, sampling multiple reconstructions and considering the uncertainty."
	- ![](../../_media/Diffusion_Application_Chung_MRI_limitations.png 'Fig. 11, Limitations: OOD recon and extreme case recon. Origin: Chung et al., 2022.')



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

[^7]: Parisi G. Correlation functions and computer simulations[J]. Nuclear Physics B, 1981, 180(3): 378-384.

[^8]: Zbontar J, Knoll F, Sriram A, et al. fastMRI: An open dataset and benchmarks for accelerated MRI[J]. arXiv preprint arXiv:1811.08839, 2018.

[^9]: Brock A, Donahue J, Simonyan K. Large scale GAN training for high fidelity natural image synthesis[J]. arXiv preprint arXiv:1809.11096, 2018.

[^10]: Zhang R. Making convolutional networks shift-invariant again[C]//International conference on machine learning. PMLR, 2019: 7324-7334.

[^11]: Tancik M, Srinivasan P, Mildenhall B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[J]. Advances in Neural Information Processing Systems, 2020, 33: 7537-7547.

[^12]: Zhao R, Yaman B, Zhang Y, et al. fastmri+: Clinical pathology annotations for knee and brain fully sampled multi-coil mri data[J]. arXiv preprint arXiv:2109.03812, 2021.

[^13]: Chung H, Sim B, Ye J C. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12413-12422.