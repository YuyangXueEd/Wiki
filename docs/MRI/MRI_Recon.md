---
title: MRI Reconstruction
description: Collection of MRI Reconstruction works
published: true
date: 2022-03-14T15:34:32.642Z
tags: mri, reconstruction, medical imaging, computer vision
editor: markdown
dateCreated: 2022-02-07T07:37:23.827Z
---

> This collection is about the MRI Reconstruction works, focusing on deep learning.

# Reviews

1. AI-Based Reconstruction for Fast MRI—A Systematic Review and Meta-Analysis [^1], 2022

# Introduction

The slow acquisition of magnetic resonance (MR) images presents a significant inconvenience to patients and healthcare systems alike.

MR data are acquired in the k-space. The k-space is related to the image domain via the Fourier transform and represents the spatial frequency information. During MRI acquisition, measures in the k-space are taken sequentially rather than simultaneously, thus prolonging the scanning time.

To address this limitation, the k-space can be undersampled. Compressed Sensing (CS) yields an aggressive acceleration rate up to 12.5-fold. CS extrapolates unknown k-space signals from existing ones, akin to image super-resolution techniques that increase image resolution by reconstructing high-frequency image details.

## Standard

### Data

1. Dataset: What the dataset was, how it was collected, selection of subsets if appropriate
2. Region: Which regions of the body the images in the dataset covered
3. MRI sequence: What the MRI sequence was, $T_1, T_2$, etc.
4. Ground truth: What the ground truth was and how it was generated
5. Partition： How the dataset was partitioned into training, validation, and testing subsets in terms of number of images， patients or MR scans
6. Augmentation: How the training dataset was augmented
7. Clinical feature: Whether the dataset contains images of pathology

### Model

1. Category: What category the model belonged to, i.e., unrolled optimization, end-to-end, or reference-driven
2. Architecture: What the structure of the model was, e.g., U-Net, DC-CNN like etc
3. Channel number: How many coils the inputs signals to the model were
4. Channel merging: If the method uses multicoil input, i.e., a parallel imaging method, how the multicoil data was merged to produce a single final reconstructed image
5. Input domain: Whether the model was designed to process raw k-space data, magnitude or complex image space data
6. Dimension: What image dimension the model was designed to process, e.g., 2D or 3D spatial, or 2D spatial-temporal
7. Input size: What size of the input MR images was
8. Loss： Which loss function were used to train the model
9. Optimizer: Which optimizer was used to update and optimise network parameters
10. Open Source: source ocde
11. Platform: which deep-learning library was used to build the model

### Evaluation method

1. Mask: What patterns of undersampling mask was used, e.g., radial, variable density
2. Acceleration: Under which acceleration ratios the model reconstructed the undersampled images
3. Comparison: What other algorithms were used to compare the performance of the  proposed algorithm
4. Metric: What quantitative and qualitative metrics were used to evluate reconstruction accuracy, e.g., NMSE, PSNR, and SSIM
5. Testing mode: Whether the model was tested prospectively, retrospectively or an established compressed sensing or both

### Result

1. Result: What were the quantitative data of performance metrics and/or qualitative comparison of representative reconstructed images
2. Computation time: Computation time in seconds on a GPU per reconstructed image

### Discussion

1. Novelty: What aspects of the model were novel, i.e., not previously reported
2. Strength: What problem in particular the model was designed to address
3. Limitation: What problem remained in the model

## META-Analysis Method

### Literature Collection

#### Four platforms:

* Google Scholar
* PubMed
* IEEE
* Crossref

#### Keywords

* MRI
* Reconstruction
* Deep Learning

### Data Analysis


# Datasets

| Public Dataset                  | Sample size | Pathology | Region      |
|---------------------------------|-------------|-----------|-------------|
| [SRI24](https://www.nitrc.org/projects/sri24/)                           | 24          | No        | Brain       |
| [MRBrainS13](https://mrbrains13.isi.uu.nl/the-mrbrains13-workshop/)                      | 20          | Yes       | Brain       |
| [ADNI](https://adni.loni.usc.edu/)                            |             | Yes       | Brain       |
| [NeoBrainS](https://neobrains.org/)                       |             |           | Brain       |
| [IXI](https://brain-development.org/ixi-dataset/)                             | 600         | No        | Brain       |
| [fastMRI](https://fastmri.org/)                         | 8400        |           | Brain, Knee |
| [Brainweb](https://brainweb.bic.mni.mcgill.ca/brainweb/)                        |             | Yes       | Brain       |
| [mridata](http://www.mridata.org/)                         | 247         |           | Knee        |
| [Caglary Campinas](https://www.ccdataset.com/)                | 212         |           | Brain       |
| [BRATS](https://www.med.upenn.edu/cbica/brats2021/)                           | 300         | Yes       | Brain       |
| [Dynamic MRI of speech movements](https://pubmed.ncbi.nlm.nih.gov/30390319/) |             |           | Brain, Neck |
| [MICCAI](https://masi.vuse.vanderbilt.edu/submission/leaderboard.html)                          | 47          | No        | Brain       |
| [HCP](https://www.humanconnectome.org/study/hcp-young-adult/data-releases)                             | 9835        | Yes       | Brain       |
| [Aggarwal 2020](https://pubmed.ncbi.nlm.nih.gov/33613806/)                   | 10          |           | Brain       |
| [Aggarwal 2019](https://ieeexplore.ieee.org/document/8434321)                   | 5           |           | Brain       |
| OAI                             |             | Yes       | Knee        |
| [Hammernik 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5902683/)                  | 100         | Yes       | Knee        |
| [Vishnevskiy 2020](https://www.semanticscholar.org/paper/Deep-Variational-Networks-with-Exponential-for-Vishnevskiy-Rau/7ee346c33879bf0ce7a03a67c6b93e260791d725#:~:text=A%20deep%20variational%20network%20is%20developed%20to%20permit,Cone-Beam%20CT%20Reconstruction%20Using%20Neural%20Ordinary%20Differential%20Equations)                | 18          | Yes       | Brain       |
| [MSChallenge](https://pubmed.ncbi.nlm.nih.gov/28087490/)                     | 35          | Yes       | Brain       |
| [CAP](https://masi.vuse.vanderbilt.edu/submission/leaderboard.html)                             | 155         |           | Heart       |
| [Liu 2020](https://europepmc.org/article/MED/31429993)                         | 31          |           | Brain       |
| [Sawyer 2013](https://core.ac.uk/display/23726142)                     | 20          | No        | Knee        |
| [MSSEG](https://www.hal.inserm.fr/inserm-01417378/file/Beaumont%20et%20al,%20MSSEG_Challenge_Proceedings.pdf)                           | 53          | Yes       | Brain       |
| MRI MS DB                       | 100         | Yes       | Brain       |
| [MIDAS](https://www.ncbi.nlm.nih.gov/sites/ppmc/articles/PMC2504702/)                           | 58          | No        | Brain       |
| [Liu 2019](https://ieeexplore.ieee.org/document/8918016)                        | 105         | No        | Brain       |

# Methods

## Deep Learning

In CS-MRI, each input node represents a pixel in the undersampled MR image. The pixels are weighted and summed to form the input for the next layer after a nonlinear activation function [^2]. The subsequent hidden layers perform a similar process to produce the final reconstructed image.

The training process is guided by the difference or error between actual and desired outcomes, described by a loss function.

The loss function computes the discrepancy between actual and desired outputs, and utilizes this information to update the parameters of the ANN that models an optimal CS-MRI reconstruction process.
 
### End-to-End

An ETE technique models the CS-MRI reconstruction process directly. In CS-MRI, the process of acquiring undersampled images is:

$$
y = U\mathrm{F}x
$$

where $x$ is the fully sampled image, $\mathrm{F}x$ is the Fourier transform of the image, i.e., its k-space representation, and $y$ is the undersampled k-space data.

ETE techniques model the inverse acquisition or reconstruction process directly, mapping from $y$ to $x$. Because of this direct mapping, the reconstruction process is usually fast.

An advantage of ETE models is that advances from other fields of deep learning are transferable to ETE designs.

One limitation is that ETE models tend to require a larger sample size to train.

ETE methods only need to optimize and update network parameters during the training procedure.

### Unrolled Optimization

UO combines deep learning with traditional iterative CS algorithms. Traditional CS techniques solve the general problem of image reconvery:

$$
\frac12 \| U\mathrm{F}x-y\|^2_2 + R(x)
$$

in which $x$ is the reconstructed image and the first term enforces the data fidelity, i.e., the reconstructed image does not differ from the undersampled one at the sampled k-space locations. The second term imposes regularization, typically sparsity constraints, on the reconstructed image to satisfy the CS criteria.

**Deep learning models are designed to learn the regularization methods to constrain image reconstruction, rather than directly modeling the reconstruction process itself**. From a Bayesian perspective, the regularizer term represents the prior knowledge about the property of the reconstructed image, e.g., sparsity.

Compared with ETE techniques, UO *incorporates prior domain knowledge* about the expected property of MR images. This reduces the solution space and facilitates model convergence and performance. It may underpin the superior performance of UO methods compared with ETE ones with fewer parameters.

In each iteration, each subnetwork has a relatively small receptive field and can perform the local transformation. This can avoid overfitting as may occur in ETE models. Compared with using the same network with the same weights in each iteration, this no-weight sharing approach has demonstrated superior performance.

Compared with ETE models, the iterative nature of UO increases the computation time.

1. Building upon this no-weight sharing approach, Zeng et al. [^3] incorporated dense connections between subnetworks, a technique inspired by image super-resolution literature.
	* This allows each subnetwork to receive the output from all the preceding subnetworks
2. one approach trains the deep-learning-based regularizer term alone without the data fidelity term [^4][^5][^6][^7]. during image reconstruction, the trained regularizer is reincorporated to optimize the reconstructed image. This solution decomposes the process of optimizing the network parameters and images into optimizing them separately.
3. A more popular approach is to train the unrolled model in an ETE fashion in a close form. The deep cascaded convolutional neural network (DC-CNN) applies the following loss function to update the network parameters and reconstructed images, $\underset{x,\theta}{\arg\max}\|U\mathrm{F}x-y\|^2_2+\lambda\|x-f(x|\theta)\|$, where $f(x|\theta)$ denotes the output of the deep learning regularizer, and $\lambda$ is a scalar to adjust the relative contributions of the two terms. 
	  * The closed form is $x = \frac{\lambda y + \mathrm{F}f(x|\theta)}{\lambda+1}$ at k-space lations that are sampled and $x = f(x|\theta)$ at k-space location that are note sampled.
    * this closed form can be interpreted as another computation layer, called the data consistency layer.
    * This cascade of neural networks becomes an ETE model and is trained in the same fashion.
    * DC-CNN is not computational efficient for parallel imaging-based CS acquisition.
    * The cascade of NN become an integral part of subsequent model designs.
    * one cannot always derive the closed form of other loss functions as trivially as in DC-CNN. Some models apply simple gradient descent, conjugate gadient descent, or auxiliary variables to implement UO in an ETE fashion.
4. ETE methods can incorporate features of UO. Various ETE models integrate the data consistency layer to enforce k-space data consistency. It enables ETE methods to enjoy the benefits of enforcing data fidelity. Therefore, combining ETE and unrolled features in a single model may increase the diversity of network designs that also share the benefits of both categories.

### Regularizer Terms Used

![regularizer_terms.png](_media/regularizer_terms.png)

* DL: Dictionary learning. Here $z$ is the latent dictionary representation or transformation of the input image. The regularizer term enforces sparsity on this dictionary representation to satisfy CS criteria. In deep dictionary learning, multiple layers of dictionaries learn this latent transform $z$.
* pFIST-SENSE： Projected fast iterative soft-thresholding algorithm-sensitivity encoding. It use a transform $\Psi$ to enforce sparsity of the reconstructed image $x$. In the deep-learning version, this transform is replaced with a neural network.
* TV: Total variation. It enforces smoothness on the reconstructed image by minimizing changes in the gradient of the image. In deep learning version, the gradiwent operator $\triangledown$ is replaced with a neural network.
* IFR-CS: Iterative feature refinement-compressed sensing. The regularizer  applies convolutional filters $k_i$ to the reconstructed image.
* SLR: Sparse and low rank approach. The regularizer minimizes the nuclear norm, i.e., the rank of the Hankel matrix $\mathcal{T}$. In the deep-learning version, theis operation is replaced with a neural network.
* SToRM: Smoothness regularization on manifolds. The regularizer has a general form $R(x)$ which can be replaced with a NN. The second term is a StoRM prior, which uses a Laplacian manifold $L$ to exploit similarity beyond local neighbor.
* Fields-of-experts: Convolutional kernels $k_i$ operate on the reconstructed image $x$, followed by trainable activation $f_i$.

### Unsupervised Learning

The objective is to minimize the difference between reconstructed images and the undersampled images at the undersampled k-space locations, i.e., enforcing data consistency [^8].

Even without fully sampled ground truth, unsupervised models can remove undersampling artifacts effectively. Even without training, a deep learning model can capture a great deal of image statistics.

Most unsupervised methods use UO and alternately optimize the reconstructed images and the model parameters [^9][^10][^11][^12].

Only one study implements an ETE training [^13].

## Network Architecture

### Variational Network

A variational network (VN), an UO methods, uses field-of-expert function as a regularizer in the image reconstruction loss function. Field-of-experts apply convolutional filters on the input undersampled images followed by activation functions, these functions are trainable.

The strength of VN is that they require 10 to 100 times fewer parameters than a typical deep-learning-based CS-MRI model.

The computational load may be lower with a smaller risk of overfitting.

### Generative Adversarial Network

With an optimal discriminator, the generator minimizes the *Shannon-Jensen divergence* between the reconstructed and fully sampled images.

* The dealiasing GAN (DAGAN) pioneers GAN-based CS-MRI, which consists of $10.8\%$ of the models. [^14][^15][^16][^17][^18][^19][^20][^21][^22]
  * It has achieved superior reconstruction performance compared with traditional CS techniques and ADMM-Net.

Disadvantages: 
 * GAN suffers from training instability, slow convergence to the global minimum, and vanishing gradient. The WGAN should mitigate these issues.
   * The WGAN minimizes the Wasserstein distance between the reconstructed and fully sampled images. WGAN-based models outperform DAGAN and cycle-GAN.
 * GAN overemphasize the high-frequency texture, thus, ignore image contents, and can produce oversmoothed appearance.
   * The LSGAN can address this problem [^14][^21]
	* While WGAN addresses the training instability of GAN, LSGAN may tackle the high frequency texture issue.
  
In all GAN-based methods, the loss functions also penalize the deviations between reconstructed and fully sampled images in the image and/or k-space domain.
 * some penalize the perceptual quality difference using the CGG 16 network
 * other enforce k-space data fidelity
 
GAN-based techniques are promising CS-MRI reconstruction methods, whose performance can be further enhanced by auxiliary penalties.

### Input Domain

$89.1\%$ of the proposed models in this reiew reconstruct the undersampled input in the image domain.

Three studies [^23][^24][^25] operate on the undersampled k-space with higher reconstruction accuracy compared with the image-domain techniques.

Two studies [^26][^27] use a hybrid of k-space and image space. For a 2-D undersampled k-space input, inverse Fourier transform was performed along the x-axis. This means that the x-axis represents the image-domain information and the y-axis k-space signals. The performance is higher over the image-domain method.

The cross-domain design leverages signals from multiple domains.

![kikinet.png](_media/kikinet.png)

* KIKI-net [^28] concatenates a subnetwork operating on the k-space (k-net) with another subnetwork on the image domain (i-net) and so on.
  * The undersampled k-space signals are first reconstructed by the k-net, followed by the inverse Fourier transform to the image domain to be processed by the i-net.
  * To satisfy hardware requirements, each subnetwork in the IKIK-net needs to be trained separately.
* Apart from k-net and i-net, one study [^29] concatenates a w-net, a subnetwork that operates on the wavelet domain of the input image.
 * The wavelet-domain network exploits both spatial and frequency features that may potentially accel- erate feature learning

The advantage of a cross-domain network is that hte k-space-based network excels in removing high-frequency artifacts. Cross-domain networks outperform networks that opearte only in the image domain.

![sub_parallel.png](_media/sub_parallel.png)

Some cross-domain networks concatenate subnetworks in parallel. The undersampled k-space signals are supplied to a k-net. In parallel, the undersampled image from the inverse Fourier transform is supplied to an 
i-net.  

### Residual Learning

Residual learning learns the difference or the residual between the ground truth and undersampled input, outperforming nonresidual learning. 

The rationale is to "constrain the generator to reconstruct only the missing details and prevent it from generating arbitrary features that may not be present in real MR images".

Residual learning can also mitigate training difficulty as the topological complexity of the residual difference may be smaller compared to the entire MR image.

### Attention

An attention module is a computational layer in the NN. This module learns the most important pixel in the input to attention, i.e., learning the optimal weights assigned to each pixel. This design achieves a higher reconstruction accuracy. 

A key limitation of attention modules is their high computational demand.

# Image Redundancy

Between 2019 and 2020, there has been a trend toward exploiting MR image redudancy aross multiple contrasts, spatiotemporal dimensions, and parallel imaging coils to improve performance and acceleration rates.

## Redundancy Across Contrast Modalities

$T_1$ weighted images provide detailed anatomical structures, while pathological features are usually more apparent in $T_2$ weighted images. To improve the clinical diagnostic power， MR images of multiple contrasts are required。 Because images with different contrasts of the same structure convey similar anatomical information， the information redundency can be used to accelerate CS-MRI.

$T_1$ weighted and $T_2$ weighted images are concatenated as a two-channel input to the deep learning model. This method achieves superior reconstruction performance compared with the model without the fully sampled $T_1$ weighted image.

Other studies concatenate undersampled images without the guidance of fully sampled ones. Alternatively, two separate networks are trained for separate contrasts with extensive crosstalk between the two networks, outperforming the same network without multicontrast information. However, the limitation of multicontrast reconstruction is that signals from one contrast may leak into another. Furthermore, the network cannot process an arbitrary number of contrasts without significant structural modifications. Despite these shortcommings, multicontrast MR reconstruction represents a significant step forward in exploiting MR image redundancy.

## Spatiotemporal Redundancy

Spatiotemporal redundancy increases in higher dimensional MR images. To illustrate, in 3-D imaging, structures in two neighboring planes are unlikely to be drastically different and are correlated. Likewise, in 4-D imaging (3D + temporal), the structures between two adjacent time frames are correlated. 

To mitigate the 3D computational demand, most studies use $2+1$ convolution. This involves a 2D convolution along two dimensions of the input image followed by a 1D convolution along the rest of the one axis. However, it is difficult to evaluate the performance of 3D deep-learning-based CS_MRI models against typical deep learning methods, most of which target 2D reconstruction. 

Hence, multidimensional MR image reconstruction tends to avoid computationally costly multidimensional convolutions.

## Parallel Imaging with Coil Redundancy

Parallel imaging is combined with CS to exploit the k-space signal redundancy collected by multiple receiver coils. 

For seperate imaging coils, many studies use separate input and outputs channels. The reconstructed images for each coil are then combined by the sum-of-squiares. One exception uses a separate network to perform the coil combination. However, neither design can handle signals of an arbitrary number of coils.

Another approach is to incoroprate parallel imaging into the optimization objective. The coil sensitivity matrix $S_i$ describes the regions that a particular coil $i$ is most sensitive to. Then, the image acquisiton model and the training objective can be modified respectively as

$$
y_i = US_iFx\\
\mathrm{and}\\
\frac12\|US_iFx-y\|^2_2 + R(x)
$$

While deep parallel imaging CS techniques can further accelerate MR acquisition, evaluating their performance against signle-coil reconstructions is challenging. This is because different datasets are required for multi- and single-coil applications. Alternatively, coil compression of raw undersampled raw undersampled multicoil data into a single coil may be used, but this comparsion may not be fair [^30].

Furthermore, in various multicoil studies, coil compression of the multicoil raw data into a smaller number of virtual coils was applied to reduce the computational demands. It is unclear whether this measure can best utilize the multicoil information or reflect the model performance on raw uncompressed multicoil signals.

# META-Analysis Results

## Dataset Characteristics

In $84.8\%$ of the studies, fully sampled MR images served as the ground truth.

The most popular dataset is human connectome projects, fastMRI, and IXI. 

The most popular augmentation techniques were flipping and rotation.  Less popular techniques included adding random noise, sharpness, contrast, and using images of different acceleration ratios. However, few studies assessed the impact of data augmentation on the performance of deep learning models in CS MRI.

![Mode and Region](_media/model_region.png)
Regarding the contrast of the MR images, T1 weighted and T2 weighted were the most popular. The least popular were MR angiography (MRA), hyperpolarised  $^{129}$Xe imaging, and contrast-enhanced MRI, probably linked to the scarcity of publicly available datasets.

Concerning the pathological features of the datasets, $26.1\%$ of the studies used pathology-free training and testing sets, while $26.1\%$ of the studies included pathology in both sets. 

## Design

### Model Architecture

Most studies use a U-Net like network, while $7.6\%$ used a structure similar to DC-CNN.  GAN-based models, the data consistency layer, and residual learning were used in a considerable proportion of the studies. Both the data consistency layer and residual learning were increasingly incorporated.

### Loss Functions

The MSE loss was the most frequently used, followed by L1 and L2 losses. Instead of MSE and L2, which can be oversmoothing, some studies chose L1 loss to facilitate convergence and produce sharper images. 

To enforce data fidelity, some studies minimized data consistency loss, i.e. the difference between the undersampled k-space data and the reconstructed k-space at the undersampled locations.

One study minimized the negative of SSIM of the reconstructed image as SSIM was a key performance metric of CS reconstruction.

Other performance metric-based loss functions included the normalized MSE, the normalized root MSE, and the mean absolute error.

For probabilistic deep learning models, the loss function was based upon maximum a posteriori or the KL divergence between the latent encodings for reconstructed and ground-truth image distribution.

Altogether, MSE was the most prevalent loss function, and recent DL based CS_MRI developments have explored the diversity of loss function choices.

The GAN-based adversarial loss was integrated to generate photorealistic images. Ablation experiments showed that each loss function was essential for DAGAN performance.

Among the optimizers that apply the gradient of the loss function to update model parameters, the most used was the Adam, with increasing popularity over time, followed by the SGD. RMSProp and gradient descent with momentum were used less frequently.

### Input Characteristics

Raw MR signals are complex numbers. One solution is to only focus on the magnitude of the complex signals. More commonly, in the input layer of the neural network, one channel processed the real part of the MR signals, the other the imaginary part. Alternatively, the two channels can be used to process the magnitude and phase of the complex number signals (no benefit).

Complex convolution[^31] convolving complex numbers using separate channels for real and imaginary images was proposed to tackle the problem of phase information of the complex signals, which exceeded the performance of normal real-valued convolution and networks that process magnitude image only.

Despite attempts to circumvent complex-valued calculations, support for complex-valued operations is still an unmet need in DL.

### Other Features

The reproducibility of DL models by considering whether the dataset and source code were accessible.

![model_clusters.png](_media/model_clusters.png)

# Evaluation Metrics

![e-metrics.png](_media/e-metrics.png)

Undersampling can be retrospective, that is, under-sample the already acquired MR images. Prospective undersampling means collecting the undersampled k-space signals directly from the MR scanners and can better reflect performance in a real-life situation. Compared with prospective undersampling, retrospective undersampling is more financially and logistically feasible.

Most studies reported the SSIM and PSNR. Fewer used NRMSE, MSE, and the signal-to-noise ratio (SNR). NRMSE and SSIM became more popular over time, whereas PSNR and NMSE decreased in popularity.

Qualitative metrics -- measures without a clearly defined mathematical expression, including the rating scores by human observers and segmentaion-based scoring -- are used less frequently. The most popular qualitative metrics were *image sharpness*, *overall quality*, *end-diastolic volume*, *ejection fraction*, and *end-systolic volume* as obtained by segmentaiton and *Likert scale*.

For low model performance, low PSNR and SSIM values, SSIM was more sensitive to changes in model performance than PSNR. In contrast, for high-performing models, PSNR was more sensitive. The result imply that PSNR and SSIM were unlikely redundant pairs of metrics as each of them maybe most sensitive to different performance levels.

Many qualitative metrics were more linearly related, including sharpness, OQ, artifact, contrast difference (CN), and the contrast-to-noise ratio (CNR). Some quantitative and qualitative metrics also correlated, including RMSE and CNR, SNR and OQ, PSNR and OQ, and SNR and sharpness.

To fully evaluate the performance of a deep learning model, most studies compared the model performance with zero-filling, which represents the baseline reconstruction results. Many studies also demonstrated the superiority of their models to other SOTAs. Typical comparison techniques included U-Net, such as architectures, DC-CNN, ADMM-Net, and DAGAN. And some traditional techniques, including DLMRI, TV, PANO, and BM3D, GRAPPA.

# Performance

We qualitified the performance of each model by the odd's ratio of improement in either SSIM or PSNR over zero-filling. 

None of the design traits was significantly linked to performance improvement. The failure of detecting significant traits may be because performance comparison among different models was confounded by the disparity of dataset and evaluation metrics among them.

Comparing the unadjusted p-values suggested that using the adam optimizer may lead to high performance, and using a U-Net, may lead to lower performance.

A higher acceleration ratio was linked to higher SSIM improvement, probably because SSIM was the most sensitive to low-performing models, and the raising acceleration ratio tended to reduce performance.

# Challenges and Outlook

Transfer learning may address the demand for large training samples.

one study shows better performance in T1 weighted images compared with FLAIR MR images, and another displays higher recon- struction errors in fat containing regions. the same deep-learning-based CS-MRI models may not display similar performance across different MR scanning sequences or anatomical regions. This is consistent with the instability of some deep-learning-based methods.

Computational challenges exist toward developing a universally applicable deep-learning-based CS-MRI algorithm.

To facilitate the future systematic review, we encourage future studies to test their model per- formance on commonly used datasets (human connectome projects, fastMRI, and IXI in Table 3) and metrics (PSNR and SSIM) and report performance on both training and testing datasets.

These developments may inspire other MRI applications, such as MRI fingerprinting, by synthesizing healthcare and personalized medicine, which will be a quantitative map of tissue properties from MR signal equipped with high throughput and low-cost imag- evolution over the signal acquisition trajectory. 


# Paper List

1. [Modl: Model-based deep learning architecture for inverse problems *Aggarwal et al. 2018*]()
2. [Learning a Variational Network for Reconstruction of Accelerated MRI Data *Hammernik et al. 2018*]()
3. [k-space deep learning for accelerated mri. IEEE transactions on medical imaging *Han et al. 2019*]()
4. [Kiki-net: cross-domain convolutional neural networks for reconstructing undersampled magnetic resonance images *Eo et al. 2018*]()
5. [Motion-Guided Physics-Based Learning for Cardiac MRI Reconstruction   *Hammernik et al. 2021*](/MRIQC/MGPBLCMRIR)
6. [Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction *Xie et al. 2022*](/MRIRecon/MDDPMUMIR)
{.links-list}

# Reference

[^1]: Y. Chen et al., ‘AI-Based Reconstruction for Fast MRI—A Systematic Review and Meta-Analysis’, Proceedings of the IEEE, vol. 110, no. 2, pp. 224–245, Feb. 2022, doi: 10.1109/JPROC.2022.3141367.

[^2]: Liberman G, Poser B A. Minimal linear networks for magnetic resonance image reconstruction[J]. Scientific reports, 2019, 9(1): 1-12.

[^3]: W. Zeng, J. Peng, S. Wang, and Q. Liu, ‘A comparative study of CNN-based super-resolution methods in MRI reconstruction and its beyond’, Signal Processing: Image Communication, vol. 81, p. 115701, Feb. 2020, doi: 10.1016/j.image.2019.115701.

[^4]: Liu Q, Yang Q, Cheng H, et al. Highly undersampled magnetic resonance imaging reconstruction using autoencoding priors[J]. Magnetic resonance in medicine, 2020, 83(1): 322-336.

[^5]: Hosseini S A H, Zhang C, Weingärtner S, et al. Accelerated coronary MRI with sRAKI: A database-free self-consistent neural network k-space reconstruction for arbitrary undersampling[J]. Plos one, 2020, 15(2): e0229418.

[^6]: Sun L, Wu Y, Fan Z, et al. A deep error correction network for compressed sensing MRI[J]. BMC Biomedical Engineering, 2020, 2(1): 1-12.

[^7]: Zhang M, Li M, Zhou J, et al. High-dimensional embedding network derived prior for compressive sensing MRI reconstruction[J]. Medical Image Analysis, 2020, 64: 101717.

[^8]: Zhao D, Zhao F, Gan Y. Reference-driven compressed sensing MR image reconstruction using deep convolutional neural networks without pre-training[J]. Sensors, 2020, 20(1): 308.

[^9]: Lewis J, Singhal V, Majumdar A. Solving inverse problems in imaging via deep dictionary learning[J]. IEEE Access, 2018, 7: 37039-37049.

[^10]: Singhal V, Majumdar A. Reconstructing multi-echo magnetic resonance images via structured deep dictionary learning[J]. Neurocomputing, 2020, 408: 135-143.

[^11]:Gong K, Han P, El Fakhri G, et al. Arterial spin labeling MR image denoising and reconstruction using unsupervised deep learning[J]. NMR in Biomedicine, 2019: e4224.

[^12]: Majumdar A. An autoencoder based formulation for compressed sensing reconstruction[J]. Magnetic resonance imaging, 2018, 52: 62-68.

[^13]: Zhao D, Zhao F, Gan Y. Reference-driven compressed sensing MR image reconstruction using deep convolutional neural networks without pre-training[J]. Sensors, 2020, 20(1): 308.

[^14]: S. U. H. Dar, M. Yurt, M. Shahdloo, M. E. Ildiz, B. Tinaz, and T. Cukur, “Prior-guided image reconstruction for accelerated multi-contrast MRI via generative adversarial networks,” IEEE J. Sel. Topics Signal Process., vol. 14, no. 6, pp. 1072–1087, Oct. 2020, doi: 10.1109/jstsp.2020.3001737.

[^15]: M. Mardani et al., “Deep generative adversarial neural networks for compressive sensing MRI,” IEEE Trans. Med. Imag., vol. 38, no. 1, pp. 167–179, Jan. 2019, doi: 10.1109/TMI.2018.2858752.

[^16]: E. Cha, H. Chung, E. Y. Kim, and J. C. Ye, “Unpaired training of deep learning tMRA for flexible spatio-temporal resolution,” IEEE Trans. Med. Imag., vol. 40, no. 1, pp. 166–179, Jan. 2021, doi: 10.1109/TMI.2020.3023620.

[^17]: V. Edupuganti, M. Mardani, S. Vasanawala, and J. Pauly, “Uncertainty quantification in deep MRI reconstruction,” IEEE Trans. Med. Imag., vol. 40, no. 1, pp. 239–250, Jan. 2021, doi: 10.1109/tmi.2020.3025065.

[^18]: M. Jiang et al., “Accelerating CS-MRI reconstruction with fine-tuning Wasserstein generative adversarial network,” IEEE Access, vol. 7, pp. 152347–152357, 2019, doi: 10.1109/ACCESS.2019.2948220.

[^19]: F. Liu, A. Samsonov, L. Chen, R. Kijowski, and L. Feng, “SANTIS: Sampling-augmented neural network with incoherent structure for MR image reconstruction,” Magn. Reson. Med., vol. 82, no. 5, pp. 1890–1904, Nov. 2019, doi: 10.1002/mrm.27827.

[^20]: G. Oh, B. Sim, H. Chung, L. Sunwoo, and J. C. Ye, “Unpaired deep learning for accelerated MRI using optimal transport driven CycleGAN,” IEEE Trans. Comput. Imag., vol. 6, pp. 1285–1296, 2020, doi: 10.1109/TCI.2020.3018562.

[^21]: T. M. Quan, T. Nguyen-Duc, and W.-K. Jeong, “Compressed sensing MRI reconstruction using a generative adversarial network with a cyclic loss,” IEEE Trans. Med. Imag., vol. 37, no. 6,
pp. 1488–1497, Jun. 2018, doi: 10.1109/TMI.2018.2820120.

[^22]: R. Shaul, I. David, O. Shitrit, and T. R. Raviv, “Subsampled brain MRI reconstruction by generative adversarial neural networks,” Med. Image Anal., vol. 65, Oct. 2020, Art. no. 101747, doi: 10.1016/j.media.2020.101747.

[^23]: M. Akçakaya, S. Moeller, S. Weingärtner, and K. Ug ̆urbil, “Scan-specific robust artificial-neural-networks for k-space interpolation (RAKI) reconstruction: Database-free deep learning for fast imaging,” Magn. Reson. Med., vol. 81, no. 1, pp. 439–453, Jan. 2019, doi: 
10.1002/mrm.27420.

[^24]: Y. Han, L. Sunwoo, and J. C. Ye, “k-Space deep learning for accelerated MRI,” IEEE Trans. Med. Imag., vol. 39, no. 2, pp. 377–386, Feb. 2020, doi: 10.1109/TMI.2019.2927101.

[^25]: B. Zhu, J. Z. Liu, S. F. Cauley, B. R. Rosen, and M. S. Rosen, “Image reconstruction by domain-transform manifold learning,” Nature, vol. 555, no. 7697, pp. 487–492, Mar. 2018, doi: 10.1038/nature25988.

[^26]: S. A. H. Hosseini et al., “Accelerated coronary MRI with sRAKI: A database-free self-consistent neural network k-space reconstruction for arbitrary undersampling,” PLoS ONE, vol. 15, no. 2, Feb. 2020, Art. no. e0229418, doi: 10.1371/journal.pone.0229418.

[^27]: T. Eo, H. Shin, Y. Jun, T. Kim, and D. Hwang, “Accelerating Cartesian MRI by domain-transform manifold learning in phase-encoding direction,” Med. Image Anal., vol. 63, Jul. 2020, Art. no. 101689, doi: 10.1016/j.media.2020.
101689.

[^28]: T. Eo, Y. Jun, T. Kim, J. Jang, H.-J. Lee, and D. Hwang, “KIKI-Net: Cross-domain convolutional neural networks for reconstructing undersampled magnetic resonance images,” Magn. Reson. Med., vol. 80, no. 5, pp. 2188–2201, Nov. 2018, doi: 10.1002/mrm.27201.

[^29]: Z. Wang, H. Jiang, H. Du, J. Xu, and B. Qiu, “IKWI-Net: A cross-domain convolutional neural network for undersampled magnetic resonance image reconstruction,” Magn. Reson. Imag., vol. 73, pp. 1–10, Nov. 2020, doi: 10.1016/j.mri. 2020.06.015.

[^30]: J. Montalt-Tordera, V. Muthurangu, A. Hauptmann, and J. A. Steeden, “Machine learning in magnetic resonance imaging: Image reconstruction,” Dec. 2020, arXiv:2012.05303. Accessed: Dec. 16, 2020.

[^31]: S. Wang et al., “DeepcomplexMRI: Exploiting deep residual network for fast parallel MR imaging with complex convolution,” Magn. Reson. Imag., vol. 68, pp. 136–147, May 2020, doi: 10.1016/j.mri.2020.02.002.