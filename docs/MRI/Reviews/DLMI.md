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

- Making it deeper
- Adversarial and attention mechanisms
	- GAN was proposed to accompany a generative model with a discriminator that tells whether a sample is from the model distribution or the data distribution.
	- Attention mechanism allows automatic discovery of "where" and "what" to focus on when describing image contents or making a holistic decision.
- Neural architecture search (NAS) and lightweight design
	- NAS aims to automatically design the architecture of a deep network for high performance geared toward to a given task.
	- Lightweight aims to design the architecture for computational efficiency on resource-constrained devices

#### Annotation Efficient Approaches

- A key idea is to leverage the power and robustness of feature representation capability derived from existing models and data even though the models or data are not necessarily from the same domain or for the same task and to adapt such representation to the task at hand.
	- including TL, domain adaptation, self-supervised learning, semisupervised learning, and weakly/partially supervised learning
	- TL aims to apply the knowledge gained via solving a source problem to a different but related target problem.
	- Domain adaption is a form of TL in which the source and target domains have the same feature space but different distributions.
		- The integrated learning mechanism offers a new possibility of dealing with multiple domains and even multiple heterogeneous tasks.
	- Self-supervised learning, a form of unsupervised learning, learns a representation through a proxy task, in which the data provide supervisory signals.
		- Once the representation is learned, it is fine-tuned by using annotated data.
	- Semi-supervised learning often trains a model using a small set of annotated images, then generates pseudo-labels for a large set of images with annotations, and learns a final model by mixing up both sets of images.
	- Weakly or partially supervised learning uses image-level annotations or weak annotations, such as dots and scribbles.
	- Unsupervised learning and disentanglement: A disentangled network structure is designed with an adversarial learning strategy that promotes the statistical matching of deep features

#### Embedding Knowledge Into Learning

Knowledge arises from various sources, such as imaging physics, statistical constraints, and task specifics and ways of embedding into a DL approach, vary too.

#### Federated Learning

To combat issues related to data privacy, data security, and data access rights, it has become increasingly important to have the capability of learning a common, robust algorithmic model through distributed computing and model aggregation strategies so that no data are transferred outside a hospital or an imaging lab.
- In contrast to conventional centralised learning with all the local data sets uploaded to one server.
- ongoing researches: reduced communication burden, data heterogeneity in various local sites, and vulnerability to attacks
- FL is applied together with domain adaptation, to train a model with boosted analysis performance and reliable discovery of disease related biomarkers.

#### Interpretability

Clinical decision-making relies heavily on evidence gathering and interpretation. 
- Lacking evidence and interpretation makes it difficult for physicians to trust the ML model's prediction.
- Interpretation is also the source of new knowledge.
- Most interpretation methods are categorised as model-based and posthoc interpretability.
	- **Model based Interpretability** is about constraining the model so that it readily provides useful information (such as sparsity and modularity) about the uncovered relationships.
		- diagnostically meaningful concepts in the latent space are encoded.
	- **Posthoc Interpretability** is about extracting information about what relationships the model has learned.

#### Uncertainty Quantification

It characterises the model prediction with confidence measure, which can be regarded as a method of posthoc interpretability. Often, the uncertainty measure is calculated along with the model prediction.

- One additional extension to uncertainty is its combination with the knowledge that the given labels are noisy.
	-  Works are now starting to emerge that take into account label uncertainty in the modeling of the network architecture and its training [^1].

## Case Studies with Progress Highlights

### Deep Learning in Thoracic Imaging (胸部成像)

Plain radiography and CT are the two most common modalities to image the chest. 
- The high contrast in density between air-filled lung parenchyma and tissue makes CT ideal for in vivo analysis of the lungs, obtaining high-quality and high-resolution images even at very low radiation dose.
- Nuclear imaging (PET) is used for diagnosing and staging oncology patients.
- MRI is somewhat limited in the lungs but can yield unique functional information.
- Ultrasound imaging is also difficult because sound waves reflect strongly at boundaries of air and tissue.

#### Segmentation of Anatomical Structures

- In 2019 and 2020, seven fully automatic methods based on U-Nets or variants thereof made the top ten for lung segmentation, and for lobe segmentation
- Both methods use a multi-resolution U-Net like architecture with several customisations.
- Segmentation of the vasculature, separated into arteries and veins, and the airway tree, including labelling of the branches and segmentation of the bronchial wall, is another important area of research.

#### Detection and Diagnosis in Chest Radiography

-  The number of publications on detecting abnormalities in the ubiquitous chest X-ray has increased enormously
	- driven by the availability of large public data sets, such as ChestXRay14, CheXpert, MIMIC, and PadChest
- Methodological contributions include novel ways of preprocessing the images, handling the label uncertainty and a large number of classes, suppressing the bones, and exploiting self-supervised learning as a way of pretraining.

#### Decision Support in Lung Cancer Screening

- In the US, screening centres have to use a reporting system called Lung-RADS. Reading lung cancer screening CT scans is time-consuming, and therefore, automating the various steps in Lung-RADS has received a lot of attention.
- The most widely studies topic is nodule detection. 
	- Lung-RADS classifies scans in categories based on the most suspicious nodule, and this is determined by the nodule type and size.
	- Lung-RADS contains the option to directly refer scans with a nodule that is highly suspicious for cancer.
- Lung-RADS directly leads to an explainable AI solution that can directly support radiologists in their reading workflow
	- One could ask a computer to directly predict if a CT scan contains an actionable lung cancer.

#### COVID-19

- Hospital used chest X-ray or CT to obtain a working diagnosis and decide whether to hospitalise patients and how to treat them.
	- The X-ray solution started from a convolutional network using local and global labels, pretrained to detect TB, fine-tuned using public and private data of patients with and without pneumonia to detect pneumonia in general, and subsequently fine-tuned on X-ray data from patients of a Dutch hospital.
	- The CT solution, called CO-RADS, aimed to automate a clinical reporting system for CT of COVID-19 suspects.
		- assesses the likelihood of COVID-19 infection on a scale from 1 (hightly unlikely) to 5 (highly likely) and quantifies the severity of disease using a score per lung lobe from 0 to 5 depending on percentage affected lung parenchyma for a maximum CT severity score of 25 points.
		- The CT severity score was derived from the segmentation results by computing the percentage of affected parenchymal tissue per lobe.

### Deep Learning in Neuroimaging

Many neuroimaging tasks, including segmentation, registration, and prediction, now have DL-based implementations. CNN have allowed for efficient network parameterisation and spatial invariance, both of which are critical when dealing with high-dimensional neuroimaging data.

The learnable feature reduction and selection capabilities of CNNs have proven effective in high-level prediction and analysis tasks and have reduced the need for highly specific domain knowledge.

#### Neuroimage Segmentation and Tissue Classification

Accurate segmentation is an important preprocessing step that informs much of the downstream analytic and predictive tasks done in neuroimaging.

- Some research has looked at improving the generalisation performance of DL-based segmentation methods across neuroimaging data sets imaged at different scanners
	- Kamnitsas et al. [^2] have proposed a training schema which leverages adversarial training to learn scanner invariant feature representations.
		- They use an adversarial network to classify the origin of the input data based on the downstream feature representation learned by the segmentation network.
		- By penalising the segmentation network for improved performance of the adversarial network, they show improved segmentation generalisation across data sets.
- The Brain Tumor Segmentation Challenge (BraTS) has provided the community with an accessible data set and a way to benchmark various approaches against one another.
- Anather task where deep networks are finding increasing success is semantic segmentation in which anatomical labels are not necessarily well-defined by image intensity changes but can be identified by relative anatomical locations.

#### Deformable Image Registration

Image registration allows for imaging analysis in a single subject across imaging modalities and time points. DL-based deformable registration with neuroimaging data has proven to be a difficult problem, especially considering the lack of ground truth.

#### Neuroimaging Prediction

By working on the raw image, reliance on complex and time-consuming preprocessing can be reduced. 

Some work has considered the use of deep Siamese networks for longitudinal image analysis. They work by jointly optimizing a set of weights on two images with respect to some distance metric between them. This setup makes them effective at identifying longitudinal changes on some chosen dimensions.

#### Use of GANs

GANs and their variants have been used in neuroimging for cross-modality synthesis, motion artifact reduction, resolution upscaling, estimating full-dosage PET images from low-dosage PET, image harmonisation, heterogeneity analysis, and more.

- The popular MedGAN [^3] propose a series of modifications and new loss function to traditional GANs, aimed at preserving anatomically relevant information and fine details.
	- They use auxiliary classifiers on the translated image to ensure that the resulting image feature representation is similar to the expected image representation for a given task.
	- They use style-transfer loss in combination with an adversarial loss to ensure that fine structures and textural details are matched in the translation.

Certain areas still lack a rigorous understanding. Important lines of research, such as learning from limited data, optimal hyperparameter selection, domain adaptation, semisupervised designs, and improving robustness, require further investigation.

### Deep Learning in Cardiovascular Imaging

There has been significant recent work in a variety of subareas of cardiovascular imaging, including image reconstruction, end-to-end learning of cardiac pathology from images, and incorporation of nonimaging information for analysis. Motion tracking and segmentation both play crucial roles in the detection and quantification of myocardial chamber dysfunction and can help in the diagnosis of cardiovascular disease (CVD). Motion tracking algorithms use segmentation results as an anatomical guide to sample points and regions of interest used to generate displacement fields.

Cardiac image segmentation is an important first step for many clinical applications. The aim is typically to segment the main chambers, LV, RV, LA, RA. This enables the quantification of parameters that describe cardiac morphology.

Cardiac motion tracking is key for deformation/strain analysis and is important for analysing the mechanical performance of heart chambers. A variety of image registration, feature-based tracking, and regularisation methods using both biomechanics models and data-driven learning have been developed.



### Deep Learning in Abdominal Imaging

### Deep Learning in Microscopy Imaging

## Discussion

### Technical Challenges Ahead

Most challenges are met by continuous improvement of solutions to the well-known data challenge. As systems are starting to be implemented across datasets, hospitals, and countries, a new spectrum of challenges is arising including *system robustness and generalisation* across acquisition protocols, machines, and hospitals.

Data preprocessing, continuous model learning, and fine-tuning across systems are a few of the new developments ahead. 

### How Do We Get New Tools Into the Clinic

In several recent works conducted to estimate the utility of AI-based technology as an aid to the radiologist, it is consistently shown that human experts with AI perform better than those without AI.

- There are a variety of reasons for this delayed clinical translation， including：
	- users being cautious regarding the technology, specifically the prospect of being replaced by AI
	- the need to prove that the technology can address real user needs and bring quantifiable benefits
	- regulatory pathways that are long and costly
	- patient safety considerations
	- economic factors, such as who will pay for AI tools

### Future Promise

One immediate step forward is to combine the image with additional clinical context, from patient record to additional clinical descriptors (blood tests, genomics, medications, vital signs, and nonimaging data such as ECG). This step will provide a transition from image space to patient-level information. Collecting cohorts will enable the population-level statistical analysis to learn about disease manifestations, treatment responses, adverse reactions from and interactions between medications.

This requires building complex infrastructure, along with the generation of new privacy and security regulations -- between hospitals and academic research institutes, across hospitals, and in multinational consortia.



## Reference

[^1]: Y. Dgani, H. Greenspan, and J. Goldberger, [Training a neural network based on unreliable human annotation of medical images](https://ieeexplore.ieee.org/document/8363518/), in Proc. IEEE 15th Int. Symp. Biomed. Imag. (ISBI), Apr. 2018, pp. 39–42.
[^2]: K. Kamnitsas et al., [Unsupervised domain adaptation in brain lesion segmentation with adversarial networks](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_47), in Proc. Int. Conf. Inf. Process. Med. Imag. Cham, Switzerland: Springer, 2017, pp. 597–609.
[^3]: K. Armanious et al., [MedGAN: Medical image translation using GANs](https://www.sciencedirect.com/science/article/pii/S0895611119300990), Comput. Med. Imag. Graph., vol. 79, Jan. 2020, Art. no. 101684.