import numpy as np
from scipy import ndimage
from typing import Tuple, Union

class NIVolume:
    def __init__(self, data, header):
        self.data = data
        self.header = header

def makehomogeneous(mag: NIVolume, datatype=None, sigma_mm=7, nbox=15):
    """
    Homogeneity correction for NIVolume from NIfTI files.

    Keyword arguments:
    - sigma_mm: sigma size for smoothing to obtain bias field. Takes NIfTI voxel size into account
    - nbox: Number of boxes in each dimension for the box-segmentation step.
    """
    if datatype is None:
        datatype = mag.data.dtype
    return makehomogeneous_inplace(datatype(mag.data), sigma=mm_to_vox(sigma_mm, mag), nbox=nbox)

def makehomogeneous_array(mag, datatype=None, sigma=None, nbox=15):
    """
    Homogeneity correction of 3D arrays. 4D volumes are corrected using the first 3D volume to
    obtain the bias field.

    Keyword arguments:
    - sigma: sigma size in voxel for each dimension for smoothing to obtain bias field. (mandatory)
    - nbox: Number of boxes in each dimension for the box-segmentation step.
    """
    if datatype is None:
        datatype = mag.dtype
    return makehomogeneous_inplace(datatype(mag), sigma=sigma, nbox=nbox)

def makehomogeneous_inplace(mag, sigma, nbox=15):
    lowpass = getsensitivity(mag, sigma=sigma, nbox=nbox)
    if np.issubdtype(mag.dtype, np.floating):
        np.divide(mag, lowpass, out=mag, where=lowpass!=0)
    else:  # Integer doesn't support NaN
        lowpass[np.isnan(lowpass) | (lowpass <= 0)] = np.finfo(lowpass.dtype).max
        mag = np.minimum(np.divide(mag, lowpass / 2048, dtype=mag.dtype), np.iinfo(mag.dtype).max)
    return mag

def getpixdim(nii: NIVolume):
    pixdim = nii.header.pixdim[1:1+nii.data.ndim]
    if np.all(pixdim == 1):
        print("Warning! All voxel dimensions are 1 in NIfTI header, maybe they are wrong.")
    return pixdim

def mm_to_vox(mm, nii_or_pixdim):
    if isinstance(nii_or_pixdim, NIVolume):
        pixdim = getpixdim(nii_or_pixdim)
    else:
        pixdim = nii_or_pixdim
    return mm / np.array(pixdim)

def getsensitivity(mag, pixdim=None, sigma_mm=None, sigma=None, nbox=15):
    """
    Calculates the bias field using the `boxsegment` approach.
    It assumes that there is a "main tissue" that is present in most areas of the object.
    """
    if isinstance(mag, NIVolume):
        return getsensitivity(mag.data, getpixdim(mag), sigma_mm=sigma_mm, nbox=nbox)
    
    if pixdim is not None and sigma_mm is not None:
        sigma = mm_to_vox(sigma_mm, pixdim)
    elif sigma is None:
        raise ValueError("Either sigma or both pixdim and sigma_mm must be provided")

    # segmentation
    firstecho = mag[:,:,:,0] if mag.ndim == 4 else mag
    mask = robustmask(firstecho)
    segmentation = boxsegment(firstecho, mask, nbox)
    
    # smoothing
    sigma1, sigma2 = getsigma(sigma)
    lowpass = gaussiansmooth3d(firstecho, sigma1, mask=segmentation, nbox=8)
    fillandsmooth(lowpass, np.mean(firstecho[mask]), sigma2)

    return lowpass

def get_default_sigma_mm(mag, pixdim):
    sigma_mm = np.zeros(min(mag.ndim, len(pixdim)))
    for i in range(len(sigma_mm)):
        sigma_mm[i] = pixdim[i] * mag.shape[i]
    sigma_mm = np.median(sigma_mm)
    sigma_mm = min(sigma_mm, 7)
    return sigma_mm

def getsigma(sigma):
    factorfinalsmoothing = 0.7
    sigma1 = np.sqrt(1 - factorfinalsmoothing**2) * sigma
    sigma2 = factorfinalsmoothing * sigma
    return sigma1, sigma2

def fillandsmooth(lowpass, stablemean, sigma2):
    lowpassmask = (lowpass < stablemean / 4) | np.isnan(lowpass) | (lowpass > 10 * stablemean)
    lowpass[lowpassmask] = 3 * stablemean
    lowpassweight = 1.2 - lowpassmask
    gaussiansmooth3d(lowpass, sigma2, weight=lowpassweight, inplace=True)

def threshold(image, mask, width=0.1):
    m = np.nanquantile(image[mask], 0.9)
    return ((1 - width) * m < image) & (image < (1 + width) * m) & mask

def boxsegment(image, mask, nbox):
    N = image.shape
    dim = image.ndim
    boxshift = np.ceil(np.array(N) / nbox).astype(int)

    segmented = np.zeros(mask.shape, dtype=np.uint8)
    for center in np.ndindex(*[range(1, N[i]+1, boxshift[i]) for i in range(dim)]):
        slices = tuple(slice(max(0, c - bs), min(c + bs, n)) for c, bs, n in zip(center, boxshift, N))
        segmented[slices] += threshold(image[slices], mask[slices])
    
    return (segmented * mask) >= 2

def robustmask(image):
    # TODO Implement robust mask creation
    pass

def gaussiansmooth3d(image, sigma, mask=None, nbox=None, weight=None, inplace=False):
    # TODO Implement 3D Gaussian smoothing
    pass