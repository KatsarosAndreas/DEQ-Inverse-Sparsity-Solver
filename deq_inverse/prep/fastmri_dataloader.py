import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, re, random, h5py, ismrmrd
from PIL import Image
from torch.utils.data import Dataset
from utils import forward_models_mri

def directory_filelist(target_directory):
    file_list = [f for f in os.listdir(target_directory)
                 if os.path.isfile(os.path.join(target_directory, f))]
    file_list = list(file_list)
    file_list = [f for f in file_list if not f.startswith('.')]
    return file_list

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def center_crop_slice(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[0]
    assert 0 < shape[1] <= data.shape[1]
    w_from = (data.shape[0] - shape[0]) // 2
    h_from = (data.shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[w_from:w_to, h_from:h_to, ...]

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

def normalize(
    data,
    mean,
    stddev,
    eps = 0.0,
):
    """
    Normalize the given tensor.
    Applies the formula (data - mean) / (stddev + eps).
    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.
    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

def normalize_instance(
    data, eps = 0.0):
    """
    Normalize the given tensor  with instance norm/
    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.
    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.
    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

class singleCoilFastMRIDataloader(Dataset):
    def __init__(self, dataset_location, transform=None, data_indices=None, sketchynormalize=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.transform = transform
        if data_indices is not None:
            filelist = directory_filelist(dataset_location)
            # print(filelist[0])
            print(len(filelist))
            try:
                self.filelist = [filelist[x] for x in data_indices]
                self.filelist[0] = 'file1002332_23.h5'
                self.filelist[0] = 'file1002444_18.h5'
            except IndexError:
                print(data_indices)
                exit()
        else:
            self.filelist = directory_filelist(dataset_location)
        self.data_directory = dataset_location
        self.fft = forward_models_mri.toKspace()
        self.ifft = forward_models_mri.fromKspace()
        self.sketchynormalize = sketchynormalize

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        filename = self.filelist[item]
        data = h5py.File(self.data_directory + filename, 'r')
        # print(str(item) + ": " + str(filename))
        # exit()w

        kspace = to_tensor(data.get('kspace').value)
        kspace_cropped = center_crop_slice(kspace, shape=[320, 320])#.permute((2,0,1))
        input_img = forward_models_mri.ifft2(kspace_cropped).permute((2,0,1))

        image_space = forward_models_mri.ifft2(kspace)
        target_img = center_crop_slice(image_space, shape=[320, 320]).permute((2,0,1))

        # image = complex_abs(image_space).permute((2,0,1))
        target_img, mean, std = normalize_instance(target_img, eps=1e-11)
        target_img = target_img.clamp(-6, 6)

        input_img, mean, std = normalize_instance(input_img, eps=1e-11)
        input_img = input_img.clamp(-6, 6)
        # if self.sketchynormalize:
            # don't ask
            # image_space *= 666.666
            # image_space *= 2000

            # image_space = image_space.clamp(min=-1, max=1)

        return input_img, target_img

class MultiSliceFastMRIDataloader(Dataset):

    def __init__(self, dataset_location, transform=None, data_indices=None, sketchynormalize=True):
  
        self.transform = transform
        self.data_directory = dataset_location
        self.fft = forward_models_mri.toKspace()
        self.ifft = forward_models_mri.fromKspace()
        self.sketchynormalize = sketchynormalize
        
        # Build list of (filename, slice_idx) pairs
        self.slice_pairs = []
        filelist = directory_filelist(dataset_location)
        filelist = [f for f in filelist if f.endswith('.h5')]
        
        for fname in filelist:
            fpath = os.path.join(dataset_location, fname)
            try:
                with h5py.File(fpath, 'r') as h5f:
                    kspace_data = h5f['kspace']
                    num_slices = kspace_data.shape[0]  # (num_slices, H, W)
                    # Restrict to knee-rich middle slices (11-27)
                    start_slice = 11
                    end_slice = min(27, num_slices - 1)  # Don't exceed available slices
                    if end_slice >= start_slice:  # Only if we have enough slices
                        for slice_idx in range(start_slice, end_slice + 1):
                            self.slice_pairs.append((fname, slice_idx))
            except Exception as e:
                print(f"Skipping {fname}: {e}")
                continue
        
        # Apply data_indices if provided (for sampling subset of slices)
        if data_indices is not None:
            self.slice_pairs = [self.slice_pairs[i] for i in data_indices]
            
    def __len__(self):
        return len(self.slice_pairs)
        
    def __getitem__(self, item):

        filename, slice_idx = self.slice_pairs[item]
        fpath = os.path.join(self.data_directory, filename)
        
        with h5py.File(fpath, 'r') as data:
            # Read single slice from multi-slice volume
            kspace_slice = data['kspace'][slice_idx]  # (H, W) complex
            kspace = to_tensor(kspace_slice)  # Convert to (H, W, 2)
            
        # EXACT same logic as original loader:
        # input_img: crop k-space first, then IFFT
        kspace_cropped = center_crop_slice(kspace, shape=[320, 320])
        input_img = forward_models_mri.ifft2(kspace_cropped).permute((2,0,1))
        
        # target_img: IFFT full k-space, then crop in image space  
        image_space = forward_models_mri.ifft2(kspace)
        target_img = center_crop_slice(image_space, shape=[320, 320]).permute((2,0,1))
        
        # EXACT same normalization
        target_img, mean, std = normalize_instance(target_img, eps=1e-11)
        target_img = target_img.clamp(-6, 6)
        
        input_img, mean, std = normalize_instance(input_img, eps=1e-11)
        input_img = input_img.clamp(-6, 6)
        
        return input_img, target_img