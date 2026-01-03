import torch, numbers, math
import torch.nn as nn
import torch.nn.functional as torchfunc
from operators.operator import LinearOperator



import numpy as np
import torch







# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def find_nearest(x, array):
    idx = (np.abs(x - array)).argmin()
    return idx

def exhaustive_sample(center_frac, acceleration, n_cols, seed):
    grid = np.linspace(-3.0,3.0,n_cols)
    sample_grid = np.zeros((n_cols,))
    num_low_freqs = int(round(n_cols * center_frac))
    pad = (n_cols - num_low_freqs + 1) // 2
    sample_grid[pad:pad+num_low_freqs] = [True]*num_low_freqs
    rng = np.random.RandomState(seed=seed)
    while True:
        sample_point = rng.standard_normal()
        if np.abs(sample_point) < 3.0:
            nearest_index = find_nearest(sample_point, grid)
            sample_grid[nearest_index] = True

        ratio_sampled = n_cols / sum(sample_grid)
        if acceleration > ratio_sampled:
            return sample_grid


def create_mask(shape, center_fraction, acceleration, seed=0, flipaxis=False):
    num_cols = shape[-2]

    # Create the mask
    mask = exhaustive_sample(center_fraction, acceleration, num_cols, seed)
    # num_low_freqs = int(round(num_cols * center_fraction))
    # prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    # rng = np.random.RandomState(seed=seed)
    #
    # mask = rng.standard_normal(size=num_cols) < prob
    # pad = (num_cols - num_low_freqs + 1) // 2
    # mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    if flipaxis:
        mask_shape[0] = num_cols
    else:
        mask_shape[-2] = num_cols
    # mask = mask.astype(np.float32)
    mask = mask.reshape(*mask_shape).astype(np.float32)
    # print(mask.shape)
    # exit()

    mask = torch.tensor(mask, requires_grad=False)
    return mask



class cartesianSingleCoilMRI(LinearOperator):
    def __init__(self, kspace_mask):
        super(cartesianSingleCoilMRI, self).__init__()
        self.register_buffer('mask', tensor=kspace_mask)

    def forward(self, input):
        input = ifftshift(input.permute((0, 2, 3, 1)))
        complex_input = torch.view_as_complex(input)
        kspace = torch.fft.fftn(complex_input, dim=1, norm="ortho")
        kspace = torch.fft.fftn(kspace, dim=2, norm="ortho")
        kspace = fftshift(kspace)
        if self.mask is not None:
            kspace_data = kspace * self.mask + 0.0
            kspace_data = ifftshift(kspace_data)
        return torch.view_as_real(kspace_data)

    def gramian(self, input):
        input = ifftshift(input.permute((0, 2, 3, 1)))
        complex_input = torch.view_as_complex(input)
        kspace = torch.fft.fftn(complex_input, dim=1, norm="ortho")
        kspace = torch.fft.fftn(kspace, dim=2, norm="ortho")
        kspace = fftshift(kspace)
        if self.mask is not None:
            kspace_data = kspace * self.mask + 0.0
            kspace_data = ifftshift(kspace_data)

        kspace_data = torch.fft.ifftn(kspace_data, dim=1, norm="ortho")
        realspace = torch.fft.ifftn(kspace_data, dim=2, norm="ortho")
        realspace = torch.view_as_real(realspace)

        output = ifftshift(realspace).permute((0,3,1,2))
        return output

    def adjoint(self, input):
        complex_input = torch.view_as_complex(input)
        complex_input = torch.fft.ifftn(complex_input, dim=1, norm="ortho")
        realspace = torch.fft.ifftn(complex_input, dim=2, norm="ortho")

        realspace = torch.view_as_real(realspace)

        output = ifftshift(realspace).permute((0, 3, 1, 2))
        return output