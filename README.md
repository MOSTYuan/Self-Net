# Self-Net

Self-Net is a deep-learning-based Python module for improving the resolution isotropy of volumetric fluorescence microscopy. This is an official CODE repository for a research article titled "Deep self-learning enables fast, high-fidelity isotropic resolution restoration for volumetric fluorescence microscopy".

## Citations

Ning K, Lu B, Wang X, et al. Deep self-learning enables fast, high-fidelity isotropic resolution restoration for volumetric fluorescence microscopy[J]. Light: Science & Applications, 2023, 12(1): 204.

## Software

- [Python 3.9 (tested on)](https://www.python.org/)
- [Conda](https://www.anaconda.com/download/)
- [Pytorch 1.13.1. (tested on)](https://pytorch.org/)
- [CUDA version 11.7 (tested on)](https://developer.nvidia.com/cuda-toolkit)
- [PyCharm 2022.3](https://www.jetbrains.com/pycharm/download/?section=windows)
- Windows 10

## Main Dependencies

- [numpy](http://www.numpy.org/) 

- [scipy](https://www.scipy.org/)

- [pytorch](https://pytorch.org/)

- [opencv](https://opencv.org/releases/)

- [scikit-image](https://scikit-image.org/)

- [tifffile](https://pypi.org/project/tifffile/)

## Hardware

- CPU or GPU that supports CUDA CuDNN and Pytorch 1.13.1.
- We tested on NVIDIA GeForce RTX 3090 (24 GB) and TITAN Xp (12 GB).

## Usage

1. For a given anisotropic 3D image stack (tiff format), first run the matlab code *'image_slice.m'* to generate lateral image slices (in the xy folder), down-sampled image slices (in the xy_lr folder), and
   axial image slices (in the xz or yz folder).

**Input parameters: raw_path, data_name, scale**



2. Run the python code *'Generate_training_data.py'* to generate file: *'train_data.npz'*.

**Input parameters: path, raw_data_path, train_data_path, signal_intensity_threshold, xy_interval, xz_interval**



3. Run *'Train_Self_net.py'* to train Self_Net.

**Input parameters: path, min_v, max_v, imshow_interval**



4. Run *'Self_net_output_volume.py'* for isotropic restoration of the raw anisotropic image stack.

**Input parameters: test_path, model_path, min_v, max_v, raw_img, scale**
