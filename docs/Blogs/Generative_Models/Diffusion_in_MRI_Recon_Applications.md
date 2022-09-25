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
	- ![](../../_media/Diffusion_Application_Chung_MRI_PC_architecture.png 'Fig. 10, Detailed Network structure of score-based model. Origin: Chung et al., 2022.')
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


#### Self-Score: Self-Supervised Learning on Score-Based Models for MRI Reconstruction [^18] 

Since previous score-based diffusion models require a large amount of fully-sampled MRI data for training, the author proposed a self-supervised method which is a fully-sampled-data-free score-based diffusion model for reconstruction. Sometimes, accessing numerous fully sampled data might be challenging, self-supervised model can learn reconstruction mappings only from undersampled data. The drawbacks are the accuracy are often inferior than supervised models, as well as poor generalisation on different sampling trajectories. However, combined with a diffusion model with good generalisation capabilities and high accuracy, self-supervised method may also has its advantages.

由于以前的 score-based diffusion model 需要大量的完全采样的MRI数据进行训练，作者提出了一种自我监督的方法，这是一种无需全采样数据的 score-based diffusion model，用于重建。有时，获取大量全采样数据可能是一种挑战，自监督模型可以只从欠采样数据中学习重建映射。其缺点是准确度往往低于监督模型，以及对不同采样轨迹的泛化性差。然而，与具有良好泛化能力和高精确度的扩散模型相结合，自监督方法也有其优势。

Used in the corrector phase of Predictor-Corrector algorithm [^3], Langevin MCMC sampling in score-based method is performed according to the score function to obtain samples that obey the distribution of data $p(x)$:

在 Predictor-Corrector 算法[^3]的修正器阶段使用，score-based 方法中的Langevin MCMC采样根据 score function 进行，以获得服从数据$p(x)$分布的样本。

$$
x_{i+1}=x_i+\frac{\eta_i}{2}s_\phi(x_i) + \sqrt{\eta_i}z_t \tag{13}
$$

where $\eta_i>0$ is the step size, and $z_t$ is standard normal distribution. The denoising score matching method first perturbs the original data $x$ to $\tilde{x}$, where $q_\epsilon(\tilde{x}|x):=\mathcal{N}(\tilde{x};x, \epsilon^2\mathbb{I})$. A reverse process using the Eq. $(13)$ above to generate the data $\tilde{x}\sim q_{\epsilon_{min}}$ from the random noise $n \sim q_{\epsilon_{\max}}(n)$. 

其中$\eta_i>0$为步长，$z_t$为标准正态分布。Denoising score matching 方法首先将原始数据$x$扰动为$\tilde{x}$，其中$q_\epsilon(\tilde{x}|x):=\mathcal{N}(\tilde{x};x, \epsilon^2\mathbb{I})$。使用上述公式$(13)$采样反向过程，从随机噪声$n\sim q_{\epsilon_{min}}$产生数据$\tilde{x}\sim q_{\epsilon_{max}}(n)$。

Focusing on MRI data $\{x_i\in \mathbb{C}^d\}^N_{i=1}$ represents the fully sampled MR images. For predicting reconstruction without fully sampled calibration data scenario, an undersampled k-space data pair is constructed by drawing on the self-supervised denoising method to construct a noisy data pair, and then the interpolation relationship between the missing data and the sampled data is learned from it to achieve missing data interpolation.

这篇文章专注于MRI数据，$\{x_i \in \mathbb{C}^d\}^N_{i=1}$ 代表完全采样的MR图像。对于没有完全采样的校准数据情况下的预测重建，通过借鉴自监督去噪方法构建一个噪声欠采样的K空间数据对，然后从中学习缺失数据和采样数据之间的插值关系，实现缺失数据插值。

The paper's main task is to accurately estimate the distribution $p(x)$ of the fully sampled image $x$ from the undersampled data $y$. The author first make an assumption that $y'$ is a subsample of undersampled measurement $y$. Suppose there exists a mapping $f_\theta$ that:

$$
f_\theta(y)=x+n_1, \ \ f_\theta(y')=\hat{y}+n_2
$$

Here is an example of an undersampling trajectory on $y$ and $y'$:

![](../../_media/Diffusion_Application_self-score_trajectory.png 'Figure. 12, Trajectories on undersampled y and subsampled y`. Orig: Cui et al., 2022.')

where $\hat{y}$ denotes the image obtained by inverse Fourier transform and channel merging of $y$, i.e., $\hat{y}=S^*F^{-1}y$, $n_1$ and $n_2$ denote the Gaussian noise with scales $\gamma_1$ and $\gamma_2$. This assumption can be considered a generalisation of the "linear interpolability" and "translation invariance" of the classical k-space interpolation method [^20] .

其中$\hat{y}$表示通过$y$的反傅里叶变换和通道合并得到的图像，即$\hat{y}=S^*F^{-1}y$，$n_1$和$n_2$表示高斯噪声的尺度$\gamma_1$和$\gamma_2$。这个假设可以被认为是经典k-空间插值方法的 "线性插值性 "和 "平移不变性 "的泛化 [^20] 。

According to the assumption above, we can obtain the conditional distribution $p(x|y, \theta)$. Using the Bayesian formula as long as we get the distribution of the parameter $\theta$:

根据上述假设，我们可以得到条件分布$p(x|y, \theta)$。使用贝叶斯公式，只要我们得到参数$\theta$的分布。

$$
p(x)=\int p(x|y, \theta)p(y)p(\theta)dyd\theta \tag{14}
$$

Since we have the undersampled k-space data pairs $D:=\{(y_i,y'_i)\}^N_{i=1}$, we can estimate $p(\theta|D)$ by Bayesian inference, i.e., by using $q(\theta|\mu_\theta, \sigma_\theta)$ to estimate $p(\theta|D)$:

由于我们有欠采样的K空间数据对$D:=\{(y_i,y'_i)\}^N_{i=1}$，我们可以通过贝叶斯推理估计$p(\theta|D)$，即用$q(\theta|\mu_\theta, \sigma_\theta)$来估计$p(\theta|D)$ 。

$$
q(\theta|\mu_\theta, \sigma_\theta) \sim \prod_s \mathcal{N}(\theta_s|\mu_{\theta, s}, \sigma_{\theta,s})
$$

The author used a Bayesian convolutional neural network (BCNN) to represent $f_\theta$ and obtain the distribution $q(\theta|\mu_\theta, \sigma_\theta)$. The undersampled k-space data pairs $\{y,y'\}$ are constructed, the BCNN is trained by minimising the KL divergence of $q(\theta|\mu_\theta, \sigma_theta)$ and $p(\theta|\{y,y'\})$. The model is directly adopted from the POCS-SPIRiT model driven neural network [^21] . The BCNN framework is shown below:

作者使用贝叶斯卷积神经网络（BCN）来表示$f/theta$并获得分布$q(\theta|\mu_\theta, \sigma_\theta)$。构建欠采样的K空间数据对$\{y,y'\}$，通过最小化$q(\theta|\mu_\theta, \sigma_\theta)$和$p(\theta|\{y,y'\})$的 KL Divergence 来训练BCN。该模型直接采用了POCS-SPIRiT模型驱动神经网络 [^21] 。BCNN的框架如下所示。

![](../../_media/Diffusion_Application_self-score_BCNN.png 'Fig. 13, The BCNN used for obtaining parameter distribution. Origin, Cui et al., 2022')

Since we now have the distribution of parameter $q(\theta|\mu_\theta, \sigma_\theta)$, we can now obtain the distribution $p(x)$. However, the ground truth score-function is not available, thus $\triangledown \log p(x)$ can be esitmated in a self-supervised manner. The minimiser of $\mathbb{E}_{p_x}[1/2\|s_\phi(x)-\triangledown \log p(x)]$ can be obtained by equivalently minimising the following objective:

由于我们现在有了参数$q(\theta|\mu_\theta, \sigma_\theta)$的分布，我们现在可以得到分布$p(x)$。然而，ground truth score-function 是无法得到的，因此$\triangledown\log p(x)$可以以自我监督的方式进行计算。$\mathbb{E}_{p_x}[1/2|s_phi(x)-\triangledown log p(x)]$的最小化可以通过等价的最小化目标函数得到。

$$
\underset{\phi}{\min}\mathbb{E}_{p(x,y,\theta)}\left[\frac12 \left\|s_\phi(x)-\frac{\partial \log p(x|\theta,y)}{\partial x}\right\|^2\right]
$$

The diffusion model is illustrated as follows. For the forward process, the model learn the score function approximating the probability density gradient of $x$ by perturbating the $f_\theta(y)$ with Gaussian noise at different scales. As for the reverse process, the model performs MCMC sampling conditional on the measurement $y$ to reconstruct MR image using the learned score function as a prior:

![](../../_media/Diffusion_Application_self-score_NCSNv2.png 'Fig. 14, NCSNv2 based diffusion model with parameter distribution. Origin: Cui et al., 2022')

The score function at different perturbation levels can be learned, and then the desired samples are obtained by performing MCMC sampling according to them. We perturb the $f_\theta(y)$ by Gaissuain noise with scales $\{\epsilon_i\}^T_{i=1}$ that satisfies $\epsilon_1 < \dots \epsilon_T$. Let $p_{\epsilon_i}(\tilde{x}|y, \theta)=\mathcal{N}(\tilde{x}|f_\theta(y), \epsilon^2_i \mathbb{I})$ and perturbed data distribution is $p_{\epsilon_i}(\tilde{x}) = \int p_{\epsilon_i}(\tilde{x}|y, \theta) p(y)p(\theta)dyd\theta$. If $\epsilon_1 = \gamma_1$, then $p_{\epsilon_1}(x)=p(x)$ holds. The training of a joint score function $s_\phi(\tilde{x}, \epsilon_i)$ is minimising the following loss:

$$
\frac{1}{2L}\sum^L_{i=1}\mathbb{E}_{p(y)q(\theta)}\mathbb{E}_{p_{\epsilon_i}(\tilde{x}|y,\theta)}\left[\left\| \epsilon_i s_phi(\tilde{x},\epsilon_i) + \frac{\tilde{x}-f_\theta(y)}{\epsilon_i}   \right|^2\right] \tag{15}
$$

The sampling process can be done with Langevin MCMC:

$$
\begin{aligned}
x_{i+1} &= x_i + \frac{\eta_i}{2} \triangledown \log p(x_i|y) + \sqrt{eta_i}z_i\\
&= x_i + \frac{\eta_i}{2}(\triangledown \log p(x_i) + \triangledown \log p(y|x_i)) + \sqrt{eta_i}z_i\\
&=x_i + \frac{\eta_i}{2}\left(s_\phi(x_i, \epsilon_i) + \frac{A^*(Ax_i-y)}{\gamma^2+\epsilon^2}\right)+\sqrt{eta_i}z_i

\end{aligned}
$$

The detailed conditional Langevin MCMC sampling is shown as follows:

![](../../_media/Diffusion_Application_self-score_CLMCMCa.png 'Fig. 15, Conditional Langevin MCMC Sampling. Origin: Cui et al., 2022.')


##### Experiments

###### Dataset

- FastMRI multi-coil knee raw data [^8] 
	- To verify the generalisability, we will test the knee data trained model's performance on brain MRI reconstruction.
- SIAT data ([中国科学院深圳先进技术研究院](https://www.siat.ac.cn/))
	- overall 1000 fully sampled multi contrast data from 10 subjects with a 3T scanner

###### Network and Training

- BCNN
	- ![](Diffusion_Application_self-score_f_theta.png 'Fig. 16, Schematic diagram of the network architecture of the $f_\theta$. Origin: Cui et al., 2022')
		- The upper and lower convolutional network modules exploit redundancies in the image domain and self-consistency in the k-space.
		- $\mathcal{P_c}$ denotes the projection onto $y'$, $\mathcal{P_c}(x)=(\mathbb{I}-P')x+y'$, where $P'$ is the undersampling pattern of $y'$.
- NCSNv2 network to learn the score function
	- $L = 50$, $\epsilon_1 = 0.0066$, number of classes is $266$, ema is true, ema rate is $0.999$.



##### Pros and Cons

###### *Pros*

- The first to propose a self-supervised learning score-based diffusion model without a fully sampled MRI training set for MRI reconstruction.
	- Evidence: "The score matching model Eq. $(15)$ works by seperating out the noise from the current image $\tilde{x}, so even if $f_\theta(y)$ is not perfectly clean image, the impact on the noise separation mechanism $(s_\phi(\cdot))$ learning is relatively small."
	- Evidence: "We utilise the prior $p(x)$ on a set of model $\theta$ over a set of data $y$ rather than the prior of a single model over a single set of data, i.e., $f_{\theta_i}(y_i), \theta_i \sim q(\theta)$ and $y_i\sim p(y)$. It has been shown that such ensemble models can often outperform single models."
- The proposed method outperforms traditional parallel imaging, self-supervised DL, and conventional supervised DL methods, and achieves comparable performance with conventional (fully sampled data trained) score-based diffusion methods.
	- Evidence: "Our proposed method performs well in aliasing pattern suppression and image texture detail recovery. In particular, it is worth mentioning that the proposed fully-sampled-data-free method outperforms the conventional supervised DL method that requires a fully sampled training dataset, which is of practical significance."
	- ![](../../_media/Diffusion_Application_self-score_results.png 'Figure. 16, Reconstruction comparison and results. Origin, Cui et al., 2022.')
-  Good genralisation ability for OOD data.
	- Evidence: "Pattern Shift: ... when the sampling patterns were inconsistent during training and testing ... we can see that both supervised and self-supervised methods degrade significantly due to the pattern shift. On the other hand, it is easy to see that our proposed method achieves satisfactory performance in both aliasing pattern suppression and detail recovery."
		- ![](../../_media/Diffusion_Application_self-score_OOD_PS.png 'Figure 17. Reconstruction results of OOD Pattern Shift. Origin: Cui et al., 2022')
	- Evidence: "Data Shift: ... when the data type (anatomies) were inconsistent during training and testing ... The proposed method in this paper can accurately reconstruct images, thus verifying its superior generalisation in data shift."
		- ![](../../_media/Diffusion_Application_self-score_OOD_DS.png 'Figure 18. Reconstruction results of OOD Data Shift. Origin: Cui et al., 2022')
- Comparable between Conventional Score-based Methods
	- Evidence: " ... we designed comparison experiments with the conventional score-based diffusion method trained on fully sampled data to verify the accuracy of the proposed self-supervised learning method on the data distribution estimation ... It can be found that the proposed self-score method performs almost identically to the score method on the fastMRI knee dataset and even slightly better than the score on the SIAT brain dataset. This experiment validates the accuracy of the proposed self-supervised learning method for data distribution estimation."
		- ![](../../_media/Diffusion_Application_self-score_comp_score.png 'Figure 19. Quantitative results between score and self-score. Origin: Cui et al., 2022')

###### *Cons*

- The proposed Assumption is a common assumption in k-space interpolation methods, there is no rigorous theory to guarantee its correctness.
- MCMC method of sampling will be more accurate for parameter $\theta$
	- Evidence: "it is also possible to collect the sample $\theta_i$ by MCMC sampling and then approximate the distribution of $\theta$ ... It is worth noting that the MCMC method and the more accurate distribution approximation methods will be reserved as our options."
- Slower than conventional diffusion model
	- Evidence: "Our method (including other score-based diffusion methods) needs to perform an iteration (MCMC sampling) to reconstruct the image, which takes a relatively long time."


### DDPM-Based Diffusion Models

#### Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction [^14] 

The authors designed a DDPM-based model since DDPM is more flexible to control the noise distribution than score-based model. The model is measurement-conditions, meaning the undersampled measurement data can directly used in training phase. Thus, no need to consider data-consistency during sampling.

作者设计了一个基于 DDPM 的模型，因为 DDPM 比 score-based model 在控制噪声分布方面更加灵活。该模型是以 measurement 为条件的，这意味着降采样的测量数据可以直接用于训练阶段。因此，在采样期间不需要考虑数据的一致性。

They defined the MRI inverse problem in a way of $y_M=MAx+\epsilon_M$, where $M$ is a diagonal matrix with sampling pattern $\Omega$, $y_M$ and $\epsilon_M$ are both vectors with $0$ at their non-sampled positions. They further defined $M^c=\mathbb{I}-M$, where $c$ means complement and $y_{M^c}=M^cAx$ which represents the non-sampled measurement. Assuming $x$ follows a distribution of $q(x)$ and given $M$, the posterior distribution, a.k.a, the undersampled reconstruction task, can be derived as:

作者以$y_M=MAx+\epsilon_M$的方式定义了MRI逆问题，其中$M$是一个对角矩阵，采样模式为$\Omega$，$y_M$和$\epsilon_M$都是在其非采样位置设为$0$的向量。进一步，他们定义了$M^c=\mathbb{I}-M$，其中$c$表示补数，$y_{M^c}=M^cAx$，表示非采样测量。假设$x$遵循$q(x)$的分布，并给定$M$，后验分布，也就是采样重建任务，可推导为：

$$
q(x|y_M,M)=\frac{q(x,y_M)|M}{q(y)}=\frac{q(y_M|x,M)q(x)}{q(y)} \tag{9}
$$

The author omits the noise $\epsilon$ here, thus $x = A^{-1}(y_M+y_{M^c})$, then the problem is transformed to estimate $q(y_{M^c}|M, y_M)$. Since the $M^c$ is complement to $M$, also as the condition, the task can be replaced to $q(y_{M^c}|M^c, y_M)$, and the reverse process can be interpreted as:

作者在这里省略了噪声$\epsilon$，因此$x=A^{-1}(y_M+y_{M^c})$，那么问题就转化为估计$q(y_{M^c}|M, y_M)$。由于$M^c$是对$M$的补，同样作为条件，任务可以替换为$q(y_{M^c}|M^c, y_M)$，则反向过程可以写为。

$$
p_\theta(y_{M^c,0}|M^c, y_M):=\int p_\theta(y_{M^c,0:T}|M^c, y_M)dy_{M^c, 1:T} \tag{10}
$$

where $y_{M^c,0}=y_{M^c}$. $p_\theta(y_{M^c,0:T}|M^c,y_M)$ is defined as:

$$
p_\theta(y_{M^c, 0:T}|M^c, y_M):=p(y_{M^c, T}|M^c, y_M)\prod^T_{t=1}p_\theta(y_{M^c, t-1}|y_{M^c,t}, M^c, y_M)
$$

It can be reparameterised by:

$$
p_\theta(y_{M^c, t-1}|y_{M^c, t}, M^c, y_M):=\mathcal{N}(\mu_\theta(y_{M^c,t},M^c,y_M), \sigma^2_t M^c)
$$

where $\sigma^2_tM^c$ is the covariance matrix and it means the noise is only added at non-sampled position because of all components of $y_{M^c},t$ at under-sampled positions are always $0$. Here we can define the forward process that Gaussian noise is gradually added to the non-sampled measurements $y_{M^c,0}$ has the following form:

$$
q(y_{M^c,1:T}|y_{M^c,0}, M^c, y_M):=\prod^T_{t=1}q(y_{M^c,t}|y_{M^c,t-1},M^c,y_M)
$$

$$
q(y_{M^c,1:T}|y_{M^c,0}, M^c, y_M):=\mathcal{N}(\alpha_t y_{M^c, t-1}, \beta^2_t M^c)
$$

Here we restrict $\alpha^2_t+\beta^2_t=1$, and let $\bar{\alpha}_t=\prod^t_{i=1}\alpha_i$, $\bar{\beta}^2_t=\sum^t_{i=1}\frac{\bar{\alpha}^2_t}{\bar{\alpha}^2_i}\beta^2_i$, and then we can derive that:

$$
q(y_{M^c,t}|y_{M^c,0},M^c, y_M)=\mathcal{N}(\bar{\alpha}_ty_{M^c,0}, \bar{\beta}^2_tM^c) \tag{11}
$$

$$
q(y_{M^c,t-1}|y_{M^c,t}, y_{M^c,0}, M, y_M)=\mathcal{N}(\tilde{\mu}_t, \tilde{\beta}^2_tM^c) \tag{12}
$$

where $\tilde{\mu}_t = \frac{\alpha_t\bar{\beta}^2_{t-1}}{\bar{\beta}^2_{t}}y_{M^c,t}+\frac{\alpha_{t-1}\bar{\beta}^2_{t}}{\bar{\beta}^2_{t}}y_{M^c,0}$, $\tilde{\beta}_t=\frac{\beta\bar{\beta}_{t-1}}{\bar{\beta}_t}$.

The training and sampling processing is very much similar to DDPM. More derivation can check the [supplement material](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-031-16446-0_62/MediaObjects/539249_1_En_62_MOESM1_ESM.pdf) by the author.

![](../../_media/Diffusion_Application_Xie_algos.png 'Fig. 10, Traning and Sampling algorithm. Origin: Xie et al. 2022.')

##### Experiments

###### Dataset

- FastMRI single-coil knee [^8] 
	- Drop the first and last five slices to avoid training the model with noise-only data
- 

###### Network and Training

- The specific design for $\epsilon_\theta(y_{M^c,t}, t, M^c, y_M)$ in the experiments is given as follows: $\epsilon_\theta(y_{M^c,t}, t, M^c, y_M)=M^c f(g(A^{-1}(y_{M^c,t}+yM),A^{-1}y_M),t;\theta)$, $f$ is a deep neural network and $g$ is the concatenation operation.
	- The author only use the magnitude of $x$, as the final image.
- - based on guided-DDPM [^15] 
	- They first use the cosine schedule, and then multiply $\beta_t$ by 0.5
	- $\beta_t$ multiplied by 0.5 so that $\bar{\beta_T}\approx 0.5$
- Even the sampling steps decrease to 250, PSNR only reduces a little
	- 20 samples average with 250 sampling steps may be a good choice

##### Pros and Cons

###### *Pros*

- The diffusion and sampling process are defined in measurement domain rather than image domain
- The diffusion process is conditioned on under-sampling mask so that data consistency is contained in the model naturally and inherently
	- Evidence: "There is no need to execute extra data consistency when sampling"
- Uncertainty: allows us to sample multiple reconstruction results from the same measurements $y$
	- Evidence: "We are able to quantify uncertainty for $q(x|y)$, such as pixel-variance."
	- Evidence: "As the acceleration factor is increased, we see that the uncertainty increases correspondingly."
- High quality of $q(x|y)$ and it outperforms baseline models and previous score-based methods
	- Evidence: "... virtually more realistic structures and less error in the zoomed-in image than ZF and U-Net"
	- ![](../../_media/Diffusion_Application_Xie_comps.png 'Figure. 11, Reconstruction results of 4x and 8x on PD data. Origin: Xie et al., 2022.')

###### *Cons*

- Slow inference
	- Evidence: "It takes 10s to generate one slice with 250 sampling steps on RTX 3090Ti."
- Only FastMRI dataset was used
- The author assume that $\epsilon_M$ is zero in its derivation
	- Evidence: "When noise is not zero, the theory, training, and inference will be more complicated but could be extended from the current method."


#### MRI Reconstruction via Data Driven Markov Chain with Joint Uncertainty Estimation [^16] 

#### Towards performant and reliable undersampled MR reconstruction via diffusion model sampling [^19] 

#### Adaptive Diffusion Priors for Accelerated MRI Reconstruction [^17] 

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

[^14]: Xie Y, Li Q. Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction[J]. arXiv preprint arXiv:2203.03623, 2022.

[^15]: Dhariwal P, Nichol A. Diffusion models beat gans on image synthesis[J]. Advances in Neural Information Processing Systems, 2021, 34: 8780-8794.

[^16]: Luo G, Heide M, Uecker M. MRI Reconstruction via Data Driven Markov Chain with Joint Uncertainty Estimation[J]. arXiv preprint arXiv:2202.01479, 2022.

[^17]: Dar S U H, Öztürk Ş, Korkmaz Y, et al. Adaptive Diffusion Priors for Accelerated MRI Reconstruction[J]. arXiv preprint arXiv:2207.05876, 2022.

[^18]: Cui Z X, Cao C, Liu S, et al. Self-Score: Self-Supervised Learning on Score-Based Models for MRI Reconstruction[J]. arXiv preprint arXiv:2209.00835, 2022.

[^19]: Peng C, Guo P, Zhou S K, et al. Towards performant and reliable undersampled MR reconstruction via diffusion model sampling[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2022: 623-633.

[^20]: Millard C, Chiew M. Self-supervised deep learning MRI reconstruction with Noisier2Noise[J]. arXiv preprint arXiv:2205.10278, 2022.

[^21]: Cui Z X, Cheng J, Zhu Q, et al. Equilibrated Zeroth-Order Unrolled Deep Networks for Accelerated MRI[J]. arXiv preprint arXiv:2112.09891, 2021.