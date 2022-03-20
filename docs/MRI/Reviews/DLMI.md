# A Review of Deep Learning in Medical Imaging: Imaging Traits, Technology Trends, Case Studies With Progress Highlights, and Future Promises

## Overview

### Traits of Medical Imaging

- Medical images have multiple modalities and are dense in pixel resolution.
	- The spatial resolution of clinical MRI has reached the submillimeter level
- Medical image data are isolated and acquired in nonstandard settings
	- Due to a lack of standardised acquisition protocols, there is a large variation in terms of equipment and scanning settings, leading to "distribution drift" phenomenon.
	- Due to patient privacy and clinical data management requirements, truly centralised open-source medical big data are rare.
- The disease patterns in medical images are numerous, and their incidence exhibits a long-tailed distribution.
	- while a small number of common diseases have sufficient observed cases for large-scale analysis, most diseases are infrequent in the clinic.
- The labels associated with medical images are sparse and noisy
	- Different tasks require different forms of annotation, which creates the phenomenon of label sparsity
	- Both interuser and intrauser labeling inconsistencies are high
- Samples are heterogeneous and imbalanced
	- The ratio between positive and negative samples is extremely uneven.
	- For example, the number of pixels belonging to a tumor is usually one to many orders of magnitude less than that of normal tissue.
- Medical image processing and analysis tasks are complex and diverse

### Clinical Needs and Applications

Medical imaging is ordered as part of a patient's follow-up to verify successful treatment.

### Key Technologies and Deep Learning

- Medical image reconstruction, which aims to form a visual representation from signals acquired by a medical imaging device.
- Medical image enhancement, which aims to adjust the intensities of an image so that the resultant image is more suitable for display or further analysis.
	- include denoising, superresolution, MR bias field correction, and image harmonization. 
- Medical image segmentation, which aims to assign labels to pixels so that the pixels with the same label form a segmented object.
- Medical image registration, which aims to align the spatial coordinates of one or more images into a common coordinate system.
- Computer-aided detection and diagnosis, aims to localise or find a bounding box that contains an object of interest.
- Other include landmark detection, image or view recognition, automatic report generation, and so on.

In mathematics, the above technologies can be regarded as function approximation methods, which approximates the true mapping $F$ that takes an image as input and outputs a specific $y, y=F(x)$.

DL has been widely used in various medical imaging tasks and has achieved substantial success in many medical imaging applications.

Assume that a training data set $\{(x_n, y_n);n=1, \dots, N\}$ is available and that a deep neural network is parameterised by $\theta$, which includes the number of layers, the number of nodes of each layer, the connecting weights, the choices of activation functions, and soon.

The neural network that is found to approximate $F$ can be written as $\phi_{\hat\theta}(x)$,  where $\hat \theta$ are the parameters that minimise the so-called loss function:

$$
L(\theta)=\frac1N \sum^N_{n=1}l(\phi_\theta(x_n), y_n) + R_1(\phi_\theta(x_n)) + R_2(\theta)
$$

where $l(\phi_\theta(x),y)$ is the itemwise loss function that pernalises the prediction error, $R_1 (\phi_\theta (x_n))$ reflects the prior belief about the output, and $R_2(\theta)$ is a regularisation term about the network parameters.

### Emerging Deep Learning Approaches

#### Network Architectures

