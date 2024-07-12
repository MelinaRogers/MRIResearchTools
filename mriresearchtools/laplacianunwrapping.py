import numpy as np
from scipy.fft import dct, idct, fft, ifft
from scipy.ndimage import convolve

def laplacianunwrap(phi):
    return laplacianunwrap_inplace(np.copy(phi))

def laplacianunwrap_inplace(phi):
    phi += 2 * np.pi * k(phi)
    return phi

# Schofield and Zhu 2003, https://doi.org/10.1364/OL.28.001194
def k(phi_w):
    return 1 / (2 * np.pi) * inverse_laplacian(laplacian_nw(phi_w) - laplacian(phi_w))

def laplacian(x):
    return -(2 * np.pi)**x.ndim / x.size * idct(pqterm(x.shape) * dct(x, norm='ortho'), norm='ortho')

def inverse_laplacian(x):
    return -x.size / (2 * np.pi)**x.ndim * idct(dct(x, norm='ortho') / pqterm(x.shape), norm='ortho')

def laplacian_nw(phi_w):
    return np.cos(phi_w) * laplacian(np.sin(phi_w)) - np.sin(phi_w) * laplacian(np.cos(phi_w))

def pqterm(shape):
    if len(shape) == 1:
        return np.arange(1, shape[0] + 1)**2
    elif len(shape) == 2:
        p, q = np.meshgrid(np.arange(1, shape[0] + 1), np.arange(1, shape[1] + 1), indexing='ij')
        return p**2 + q**2
    elif len(shape) == 3:
        p, q, t = np.meshgrid(np.arange(1, shape[0] + 1), np.arange(1, shape[1] + 1), np.arange(1, shape[2] + 1), indexing='ij')
        return p**2 + q**2 + t**2
    elif len(shape) == 4:
        p, q, t, r = np.meshgrid(np.arange(1, shape[0] + 1), np.arange(1, shape[1] + 1), np.arange(1, shape[2] + 1), np.arange(1, shape[3] + 1), indexing='ij')
        return p**2 + q**2 + t**2 + r**2
    else:
        raise ValueError("Unsupported number of dimensions")

def laplacianunwrap_fft(phi, z_weight=1):
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, :] = [z_weight, -2 * (1 + z_weight), z_weight]
    kernel[1, :, 1] += [-1, 2, -1]
    kernel[:, 1, 1] += [-1, 2, -1]

    def laplacian(x):
        return convolve(x, kernel, mode='wrap')

    kernel_full = np.zeros_like(phi)
    s = kernel.shape
    kernel_full[:s[0], :s[1], :s[2]] = kernel
    kernel_full = np.fft.fftshift(kernel_full)

    del_op = np.fft.fftn(kernel_full)
    del_inv = 1 / del_op
    del_inv[~np.isfinite(del_inv)] = 0

    del_phase = np.cos(phi) * laplacian(np.sin(phi)) - np.sin(phi) * laplacian(np.cos(phi))
    unwrapped = np.real(np.fft.ifftn(np.fft.fftn(del_phase) * del_inv))

    return unwrapped

def laplacianunwrap_mixed(phi):
    kernel = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 1, 0],
                        [1, -6, 1],
                        [0, 1, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]])

    def laplacian(x):
        return convolve(x, kernel, mode='wrap')

    del_phase = np.cos(phi) * laplacian(np.sin(phi)) - np.sin(phi) * laplacian(np.cos(phi))
    unwrapped = inverse_laplacian(del_phase)

    return unwrapped