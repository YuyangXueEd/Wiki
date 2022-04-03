# fastMRI: An Open Dataset and benchmarks for Accelerated MRI

## Introduction

In this work we describe the first large-scale release of raw MRI data that includes $8344$ volumes, consisting of $167, 375$ slices, associated with *in vivo* examinations from a range of MRI systems. In addition, we are releasing processed MR images in DICOM format from $20,000$ knee and brain examinations from a representative clinical patient population, consisting of more than $1.57$ million slices.

## Introduction to MR Image Acquisition and Reconstruction

The spatially-resolved image $m$ can be estimated from the full k-space $y$ by performing an inverse multidimensional Fourier transform:

$$
\hat{m}=\mathcal{F}^{-1}(y)
$$

where $\hat{m}$ is a noise-corrupted estimate of the true image $m$. The number of samples captured in k-space is a limiting factor for the speed of MR imaging. Fewer samples can be captured by sampling up to a lower maximum frequency, however this produces images of lower spatial resolution. An alternative undersampling approach involves omitting some number of k-space samples within a given maximum frequency range, which then results in aliasing artifacts.

### Parallel MR Imaging

In parallel MR imaging, multiple receiver coils are used, each of which produces a separate k-space measurement matrix. Each of these matrices is different, since the view each coil provides of the imaged volume is modulated by the differential sensitivity that coil exhibits to MR signal arising from different regions; Each coil measures Fourier components of the imaged volume multiplied by a complex-valued position-dependent coil sensitivity map $S_i$. The measured k-space signal $y_i$ for coil $i$ in an array of $n_c$ coils is given by:

$$
y_i=\mathcal{F}(S_im) + noise
$$

where the multiplication is entry-wise.

![MMRI_Recon](../../_media/MMRI_recon.png)

This image shows the absolute value of the inverse *discrete Fourier transform* (DFT) of fully-sampled complex-valued k-space signals for each coil in a 15-element coil array.

If the sensitivity maps are known, and the k-space sampling is full, then the set of linear relations between $m$ and each $y_i$ defines a linear system that is overdetermined by a factor of $n_c$. It may be inverted using a pseudoinverse operation to produce a reconstruction of $m$, as long as the linear system is full rank. The quality of this reconstruction will depend on the measurement noise, since the signal-to-noise ratio is poor in parts of the volume where the coil sensitivity is low.

In accelerated parallel imaging, each coil's k-space signal is undersampled. As long as the total number of measurements across all coils exceeds the number of image voxels to be reconstructed, an unregularised least squares solution can still be used, leading to a theoretical $n_c$-fold speedup over fully-sampled single-coil imaging. Each extra coil effectively produces an additional "sensitivity-encoded" measurement of the volume, which augments the frequency and phase encoded measurements obtained from the sequential application of magnetic field gradients in the MR pulse sequence.

In practice, the use of sub-sampling results in significant amplification of noise, and regularisation is usually needed. As a result, in the clinic, parallel imaging acceleration factors are typically on the order of two to three.

### Machine Learning Reconstruction of Undersampled MRI Data

In the machine learning approach, a reconstruction function:

$$
\hat{m} = B(y)
$$

is learned from input and output pair tuples $(y,m)$ drawn from a population. The goal is to find a function $B$ that minimises the risk over the population distribution:

$$
B^* = \underset{B}{\arg \min}R(B)\ \ \mathrm{where}\ R(B)=\mathbb{E}_{(y, m)}[L(B(y), m)]
$$

In practice this optimisation problem must be approximated with the empirical risk using a sample $\{(m^{(i)}, y^{(i)})\}^{n_{data}}_{i=1}$ from the population, with respect to a loss function $L$:

$$
R_{empirical}(B)=\frac{1}{n_{data}}\sum^{n_{data}}_{i=1}L(B(y^{(i)}), m^{(i)})
$$

## Prior Public Datasets

See [A list of Public MRI Dataset](Dataset_List.md)

## The fastMRI Dataset and Associated Tasks

Four types of data:

1. **Raw multi-coil k-space data**: unprocessed complex-valued multi-coil MR measurements
2. **Emulated single-coil k-space data**: combined k-space data derived from multi-coil k-space data in such as way as to approximate single-coil acquisitions, for evaluation of single-coil reconstruction algorithms.
3. **Ground-truth images**: real-valued images reconstructed from fully-sampled multi-coil acquisitions using the simple root-sum-of-squares method detailed below.
4. **DICOM images**: spatially-resolved images for which the raw data was discarded during the acquisition process. These images are provided to represent a larger variety of machine s and settings than are present in the raw data.

Two distinct types of tasks:

1. **Single-coil reconstruction task**: reconstruct images approximating the ground-truth from undersampled single-coil data
2. **Multi-coil reconstruction task**: reconstruct images approximating the ground-truth from undersampled multi-coil data.

For each task we provide an official split of the k-space data and ground-truth images into *training* and *validation* subsets that contain fully-sampled acquisitions, as well as *test* and *challenge* subsets which contain k-space data that have been subjected to undersampling masks. During training of a machine-learning model, the training k-space data should be programmatically masked following the same procedure.

The rationale for having a single-coil reconstruction task is two fold:
1. to lower the barrier of entry for researchers who may not be familiar with MRI data, since the use of a single coil removes a layer of complexity
2. to include a task that is relevant for the single-coil MRI machines still in use throughout the world

The DICOM images may be useful as additional data for training. Their distribution is different from that of the ground-truth images, since they were acquired with a larger diversity of scanners, manners of acquisition, reconstruction methods, and post-processing algorithms, so the application of transfer-learning techniques may be necessary. Most DICOM images are the result of accelerated parallel imaging acquisitions and corresponding reconstructions, with image quality that differs from that of putative fully-sampled acquisitions and reconstructions. The ground-truth images may represent a higher standard of image quality than the clinical gold standard, for which full sampling is not routine or even practical.

### Anonymisation

Raw data was anonymised via conversion to the vendor-neutral ISMRMRD [^1] format, with manual checking of metadata in raw data files, as well as spot checking of all metadata and image content.

DICOM data was anonymised using the RSNA clinical trial. processor. Each DICOM image is performed manual inspection for the presence of unexpected protected health information (PHI).

### Knee k-space Data

Multi-coil raw data was stored for $1,595$ scans acquired for the purpose of diagnostic knee mRI. For each scan, a single fully sampled MRI volume was acquired on one of three clinical 3T system or one clinical 1.5T system. Data acquisition used a 15 channel knee coil array and conventional Cartesian 2D TSE protocal employed clinically.

The dataset includes data from two pulse sequences, yielding coronal proton-density weighting with and without fat suppression.

![wio_fat_suppression.png](../../_media/wio_fat_suppression.png)

The following sequence parameters are used:
- Echo train length $4$
- matrix size $320 \times 320$
- in-plane resolution $0.5$mm$\times 0.5$ mm
- slice thickness $3$mm
- no gap between slices
- repetition time (TR) ranging between $2200$ and $3000$ ms
- echo time (ET) between $27$ and $34$ ms

### Brain k-space Data

Data from %6970$ fully sampled brain MRIs were obtained using $11$ magnets across $5$ clinical locations using $1.5T$ and $3T$ field strengths.

The raw dataset includes axial T1 weighted, T2 weighted and FLAIR images. Some of the T1 weighted acquisitions included admissions of contrast agent (labelled T1 POST).

![brain_MRI.png](../../_media/brain_MRI.png)

To ensure data de-identification, we used only axial 2D images in this dataset. We used zero matrices to replace the k-space slices $\underset{\approx}{>} 5$mm below the orbital rim. All processed k-spaces were then reconstructed to images in DICOM format, loaded into a picture archival communication system (PACS) and all images were visually checked by certified MR technologists to confirm exclusion of identifying facial features.

### Knee emulated Single-coil k-space Data

We used an emulated single-coil (ESC) methodology to simulate single-coil data from a multi-coil acquisition.

ESC computes a complex-valued linear combination of the responses from multiple coils, with the linear combination fitted to the ground-truth root-sum-of-squares reconstruction in the least-squares sense.

### Knee DICOM Data

In addition to the scanner raw data described above, the fastMRI dataset includes DICOM data from $10,000$ clinical knee MRI scans. These images represent a wider variety of scanners and pulse sequences than those represented in the collection of raw data.

five clinical pulse sequences:

1. Coronal proton-density weighting without fat suppression,
2. Coronal proton-density weighting with fat suppression,
3. Sagittal proton-density weighting without fat suppression,
4. Sagittal T2 weighting with fat suppression, and
5. Axial T2 weighting with fat suppression.

The Fourier transformation of an image from a DICOM file does not directly correspond to the originally measured raw data，due to the inclusion of additional post-processing steps in the vendor-specific reconstruction pipeline. Most of the DICOM images are also derived from accelerated acquisitions and are reconstructed with parallel imaging algorithms, since this baseline acceleration represents the current clinical standard. **The image quality of DICOM images, therefore, is not equivalent to that of the ground truth images directly associated with fully sampled raw data.**

### Brain DICOM

$10,000$ brain MRI DICOM studies are also included.

Axial 2D image volumes are included with: T1, T2, and T2 FLAIR. All studies represent unique individuals and there is no subject overlap with the brain raw data.

Finally, we present $10,000$ brain MRI DICOM studies from $10,000$ unique subjects, each one including axial 2D DICOM image volumes through the majority of the brain representing a broad range of neurological pathologies.

### Ground Truth

The root-sum-of-squares reconstruction method applied to the fully sampled k-space data provides the ground truth for the multi-coil dataset. The single-coil dataset includes two ground truth reconstructions, which we denote ESC and RSS:
- The ESC ground truth is given by the inverse Fourier transform of the single-coil data
- The RSS ground truth is given by the root-sum-of-squares reconstruction computed on the multi-coil data that were used to generate the virtual single-coil k-space data.
All ground truth images are cropped to the central $320 \times 320$ pixel region to compensate for readout-direction oversampling that is standard in clinical MR examinations.

RSS is one of the most commonly-used coil combination methods in clinical imaging. It first applies the inverse Fourier Transform to the k-space data from each coil:

$$
\tilde{m}_i=\mathcal{F}^{-1}(y_i)
$$

where $y_i$ is the k-space data from the $i$th coil and $\tilde{m}_i$ is the $i$th coil image. Then, the individual coil images are combined voxel by voxel as follows:

$$
\tilde{m}_{rss}=(\sum^{n_c}_{i=0}|\tilde{m}_i|^2)^{-1/2}
$$

where $\tilde{m}_{rss}$ is the final image estimate and $n_c$ is the number of coils. The RSS image estimate is known to converge to the optimal, unbiased estimate of $m$ in the high-SNR limit.

### Dataset Split

![fastMRI_count.png](../../_media/fastMRI_count.png)

### Cartesian Undersampling

The undersampling is performed by retrospectively masking k-space lines from a fully-sampled acquisition. k-space lines are omitted only in the phase encoding direction, so as to simulate physically realisable accelerations in 2D data acquisitions.The same undersampling mask is applied to all slices in a volume, with each case consisting of a single volume. The overall acceleration factor is set randomly to either four or eight, with equal probability for each.

When the acceleration factor equals four, the fully-sampled central region includes 8% of all k-space lines; when it equals eight, 4% of all k-space lines are included.

![fastMRI_undersample.png](../../_media/fastMRI_undersample.png)

## Metrics












## Reference

[1]: Souheil J Inati, Joseph D Naegele, Nicholas R Zwart, Vinai Roopchansingh, Martin J Lizak, David C Hansen, Chia-Ying Liu, David Atkinson, Peter Kellman, Sebastian Kozerke, et al. [ISMRM raw data format: a proposed standard for MRI raw datasets](https://www.opensourceimaging.org/project/ismrmrd). Magnetic resonance in medicine, 77(1), 2017.