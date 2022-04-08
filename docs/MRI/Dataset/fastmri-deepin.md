# Deep Into FastMRI Dataset

This article is designed to provide insight into the data details and structure of the fastMRI [^1] dataset, as well as a parsing of the data reads and baseline model of the [fastMRI](https://github.com/facebookresearch/fastMRI) library.

## Data Description

### Single-coil Knee

The dataset includes both raw MRI k-space data and magnitude DICOM images [^2].
- The k-space data comprises $1594$ measurement datasets obtained in knee mRI examinations from a range of MRI systems and clinical patient populations, with responding images derived from the k-space data using reference image reconstruction algorithms.
- The DICOM data represent an additional $10012$ clinical image datasets from $9290$ patients undergoing similar knee MRI examinations.

 Five sequences for different contrasts and image orientations:
 1. coronal proton density weighted
 2. coronal proton density weighted with fat suppression
 3. axial T2 weighted with fat suppression
 4. sagittal proton density weighted, and 
 5. sagittal T2 weighted with fat suppression.

The k-space dataset only contains the coronal acquisitions.

![Acquisition parameters](../../_media/fastMRI_knee_kspace_acq_params.png)

The DICOM dataset contains data from all five sequences, whose parameters can be found directly in the headers of the data.

The k-space data were deidentified via conversion to the vendor-neutral ISMRM raw data format.

DICOM data were deidentified by using the RSNA clinical Trial Processor tool. 

 The k-space is split up into the following files for download:
1. `singlecoil_train` (88 GB)
2. `singlecoil_val` (19 GB)
3. `singlecoil_test` (7 GB)
   
The total size of the combined DICOM image files is approximately 164 GB, which are stored with lossless JPEG 2000 image compression:
1. DICOMs_batch1 (134 GB)  
2. DICOMs_batch2 (30 GB).

#### k-space dataset

Fully sampled k-space data from 1594 consecutive clinical MRI proton density–weighted acquisitions of the knee in the coronal plane with and without  
frequency-selective fat saturation are included.

Scans were performed on three clinical `3-T` systems (*Siemens Magnetom Skyra, Prisma,* and *Biograph-mMR*) and one clinical `1.5-T` system (*Siemens Magnetom Aera*) using clinical multichannel receive coils.

Example images from reference reconstructions are shown in the figure below.

![fastMRI_knee_wio_fat_sup.png](../../_media/fastMRI_knee_wio_fat_sup.png)

The data are provided together with metadata that allow reconstruction of images by means of a simple inverse Fourier transform. In particular, the individual k-space lines are already correctly sorted accord- ing to their position in the acquisition trajectory. No further preprocessing steps were performed on the data.

![fastMRI_knee_kspace_metadata.png](../../_media/fastMRI_knee_kspace_metadata.png)

Because the data were acquired with a multichannel receive array coil, a proper combination of the individual coil images is a necessary step in the image reconstruction process. The most straightforward approach is to use a *sum-of-squares* combination of the individual coil images.

Image reconstruction of accelerated acquisitions via parallel imaging requires an additional calibration step to obtain coil sensitivity information: either obtaining maps of the coil sensitivity profiles or estimating convolution kernels in k-space.

We also provide simulated single-coil k-space data derived from the acquired multi-coil k-space data using an “emulated single-coil” combination algorithm.

The rationale for providing simulated single-coil data—even though reconstruction from multi-coil data:

1. To lower the barrier of entry for researchers who may not be familiar with MRI data, since the use of a single coil removes a layer of complexity
2. to include a task that is relevant for the single-coil MRI machines still in use through- out the world, and
3. to separate out the aspects of reconstruction related to compressed sensing rather than parallel imaging.

Examples in the test and challenge sets contain undersampled k-space data. The undersampling is performed by retrospectively masking k-space lines from a fully sampled acquisition. k-space lines are omitted only in the phase-encoding direction to simulate physically realisable accelerations in 2D data acquisitions. The undersampling mask is chosen randomly for each example, subject to constraints on the number of fully sampled central lines and the overall undersampling factor.

![fastMRI_masks.png](../../_media/fastMRI_masks.png)

#### HDF5 File Details

##### Keys

- ***ismrmrd_header***: The XML header from the `ISMRMRD` file that was used to generate the `HDF5` file. Here is [an example](https://gist.github.com/YuyangXueEd/6f1457a788309dcdf7746ef7e2377c5c) of the `ISMRMRD` header from one train file. Here are some important keys:
	- *systemFieldStrength_T*: `1.5T` or `3T`
	- *encoding - encodedSpace - fieldOfView_mm*: Field of view
	- *encoding - encodedSpace - matrixSize*: Crop size
	- *encoding - encodingLimits - slice*: How many slices
	- *encoding - reconSpace - fieldofView_mm*: Field of view in mm
	- *encoding - reconSpace - matrixSize*: Actual reconstruction size
	- *encoding - trajectory*: Cartesian or non-Cartesian
- ***kspace***: `complex64` Numpy data, using `to_tensor()` to convert to PyTorch tensor
- ***reconstruction_esc***: a reconstructed output using ESC algorithm
- ***reconstruction_rss***: a reconstructed output using RSS algorithm

##### Attributes

- ***acquisition***:Acuisition protocol.
	- *Knee*: `CORPD` or `CORPDF`
- ***max***: The largest entry of the target volume. Not shown in test or challenge files.
- ***norm***: The Euclidean norm of the target volume. Not shown in test or challenge files.
- ***patient_id***: A unique string identifying the examination.
- ***acceleration***: Acceleration factor of the undersampled k-space trajectory (either 4 or 8). Only available in the test dataset.
- ***num low frequency***: The number of low-frequency k-space lines in the undersampled k-space trajectory. This attribute is only available in the test dataset.

## Data Manipulation

### `fastmri.data.mri_data`

#### `et_query`

This function can be used to query a xml document via `ElementTree`, and uses `qlist` for nested queries. We can use this to read `hdf5` files for each sample.

#### `fetch_dir`

Set `data_config_file` for the project. Simply overwrite the variables for `knee_path` and `brain_path` and this function will retrieve the requested subsplit of the data for use.


#### `CombinedSliceDataset`

A container for combining slice datasets.

Parameters:
- *roots*: paths to the datasets
- *challenges*: `singlecoil` or `multicoil`
- *transforms*: A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function should take `kspace`, `target`, `attributes`, `filename`, and `slice` as inputs.
- *sample_rates*: A sequence of floats between $0$ and $1$. This controls what fraction of the slices should be loaded. Sample by slices.
- *volume_sample_rates*: A sequence of floats between $0$ and $1$. This controls what fraction of the volumes should be loaded. Sample by volumes.
- *use_dataset_cache*: Whether to cache dataset metadata. Useful for large datasets like the brain data.
- *dataset_cache_file*:A file in which to cache dataset information for faster load times.
- *num_cols*: if provided, only slices with the desired number of columns will be considered.

The Lengths of *roots*, *transforms*, *challenges*, *sample_rates* should be matched.

#### `SliceDataset`

A PyTorch Dataset that provides access to MR image slices. The parameters are pretty much the same as `CombinedSliceDataset`. 

A static function `_retrieve_metadata` can get metadata from `hdf5` files, and it give the shape and padding information.

#### `AnnotatedSliceDataset`

A PyTorch Dataset that provides access to MR image slices with annotation. Not very useful for reconstruction tasks.

### `fastmri.data.subsample`

#### `MaskFunc`

An object for GRAPPA-style sampling masks. This creates a sampling mask that densely samples the centre while subsampling outer k-space regions based on the undersampling factor.

When called, `MaskFunc` uses internal functions to create mask by:
1. creating a mask for the k-space centre,
2. create a mask outside of the k-space centre
3. combining them into a total mask.

Parameters:
- *center_fractions*: Fraction of low-frequency columns to be retained. If multiple values are provided, then one of these number is chosen uniformly each time.
- *accelerations*: Amount of under-sampling.
- *allow_any_combination*: whether to allow cross combinations of elements from *center_fractions* and *accelerations*.
- *seed*: random number generator

Functions:

- *sample_mask*: Using `shape` and `offset` as input to return two components of a k-space mask, both the centre mask and the acceleration mask.
	- `num_low_frequencies = round(num_cols * center_fraction`
- *reshape_mask*: Reshape mask to desired output shape
- *calculate_acceleration_mask*
- *calculate_center_mask*: Using `shape` and `num_low_freqs` as input to build centre mask based on number of low frequencies.

#### `RandomMaskFunc`

To create a random sub-sampling mask of a given shape. A subclass of `MaskFunc`.

For example, if `accelerations =[4,8]` and `center_fractions = [0.08, 0.04]`, then there is a $50\%$ probability that 4-fold acceleration with $8\%$ centre fraction is selected and $50\%$ probability that 8-fold acceleration with $4\%$ centre fraction is selected.

#### `EquiSpacedMaskFunc`

Return an equally-spaced k-space mask.

#### `EquispacedMaskFractionFunct`

Equispaced mask with approximate acceleration matching.

#### `MagicMaskFunc`

Masking function for exploiting conjugate symmetry via offset-sampling.

#### `MagicMaskFractionFunc`

Similarly, this method exactly matches the target acceleration by adjusting the offsets.

### `fastmri.data.transforms`

#### `to_tensor`

Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts are stacked along the last dimension.

#### `tensor_to_complex_np`

Converts a complex torch tensor to numpy array.

#### `apply_mask`

Subsample given k-space by multiplying with a mask.

Parameters:
- *data*: the input k-space data, as least 3 dimensions, where dimensions `-3` and `-2` are the spatial dimensions, and the final dimension has size 2 (for complex values).
- *mask_func*: A function that takes a shape and a random number seed and returns a mask
- *seed*
- *padding*: Padding value to apply for mask

Returns:
- masked data: sampled k-space data
- mask: the generated mask
- num_low_frequencies

#### `mask_centre`

Initialises a mask with the centre filled in.

#### `batched_mask_center`

Initialises a mask with the centre filled in. Can operate with different masks for each batch element.

#### `center_crop`

Apply a centre crop to the input real image or batch of real images.

#### `complex_center_crop`

Apply a centre crop to the input image or batch of complex images.

#### `center_crop_to_smallest`

Apply a centre crop on the larger image to the size of the smaller.

#### `UnetSample`

A subsampled image for U-Net reconstruction.

### `fastmri.data.volume_sampler`

#### `VolumeSampler`

Sampler for volumetric MRI data. Based on PyTorch `DistributedSample`, the difference is that all instances from the same MRI volume need to go to the same node for distributed training.

`dataset` example is a list of tuples `(fname, instance)`, where `fname` is the file name.

Parameters:

- *dataset*: An MRI dataset (e.g. `SliceData`)
- *num_replicas*: Number of processes participating in distributed training.
- *rank*: Rank of the current process within `num_replicas`.
- *shuffle*
- *seed*

### `fastmri.utils`

#### `save_reconstructions`

Save reconstruction images in h5 files.

### `fastmri.losses`

#### `SSIMLoss`

### `fastmri.evaluate`

- `mse`: Compute Mean Squared Error
- `nmse`: Compute Normalised Mean Squared Error
- `psnr`: Compute Peak Signal to Noise Ratio metric 
- `ssim`: Compute Structural Similarity Index Metric
- `evaluate`: Using `Metrics` class to evaluate reconstruction images.

#### `Metrics`

### `fastmri.fftc`

- *fft2c_new*: Apply centred 2d FFT
- *ifft2c_new*: Apply centred 2d IFFT
- *fftshift*: Similar to `np.fft.fftshift` but applies to PyTorch tensors
- *ifftshift*: Similar to `np.fft.ifftshift` but applies to PyTorch tensors

## Baseline Models

### U-Net



## Reference

[^1]: J. Zbontar _et al._, ‘fastMRI: An Open Dataset and Benchmarks for Accelerated MRI, Dec. 2019, Accessed: Apr. 01, 2022. [Online]. Available: [http://arxiv.org/abs/1811.08839](http://arxiv.org/abs/1811.08839)

[^2]: F. Knoll _et al._, ‘fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning’, _Radiology: Artificial Intelligence_, vol. 2, no. 1, p. e190007, Jan. 2020, doi: [10.1148/ryai.2020190007](https://doi.org/10.1148/ryai.2020190007).