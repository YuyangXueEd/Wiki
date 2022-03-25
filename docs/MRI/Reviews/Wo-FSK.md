## Introduction

Due to physiological constraints such as organ motion or physical constraints such as signal decay, it is difficult, impractical and impossible to obtain fully sampled data. Here, the main related methods in deep learning based MRI reconstruction without fully sampled data are reviewed.

## Traditional Methods for MRI reconstruction

### Reconstruction model

A multi-coil imaging model can be expressed as follows:

$$
y = Ax + \eta\ \ \mathrm{with} \ \ A_i=UFS_i
$$

where $x \in \mathbb{C}^N$ is the image to be reconstructed, $y \in \mathbb{C}^M$ is the noisy measured data ($M << N$), $\eta$ is the noise, and $A$ denotes a measurement operator consisting of a sampling matrix $U \in \mathbb{R}^{M\times N}$, Fourier transform operator $F$ and the sensitivity map matrix $S_i$ for the $i$th coil. A common reconstruction model is to add a regularisation term to constrain its solution space,

$$
\underset{x}{\arg\min}\frac12 \|y-Ax\|^2_2 + \lambda \mathcal{R}(x)
$$

where $\|y-Ax\|^2_2$ ensures consistency with the measured data, $\mathcal{R}(x)$ is a regularisation item, and $\lambda$ is a tradeoff between the data consistency and the regularisation terms.

Multiple regularisation terms can be chosen such as 2D wavelet, total variation (TV), dictionary, 3D wavelet, 3D k-t sparse, 3D low-rank. Some methods are often used to iteratively solve the above optimisation problems. 

Sparsity or low-rankness constraints are often used as priors to reduce the artefacts of the reconstruction image when the acceleration rate is high. Researchers found that the key to MRI reconstruction based on compressed sensing lies in the design of the sparse domain, which mainly includes pre-constructed or adaptive basis and dictionary .

Low-rankness methods are mainly used for dynamic and high-dimensional imaging by exploring the relationship between multiple images.

Although good achievements have been achieved, traditional optimisation reconstruction methods complete iterations with more time.

### Optimisation Algorithms

##