import numpy as np
from scipy import ndimage
from skimage import morphology, measure

def robustmask_inplace(image, maskedvalue=None):
    if maskedvalue is None:
        maskedvalue = np.nan if np.issubdtype(image.dtype, np.floating) else 0
    mask = robustmask(image)
    image[~mask] = maskedvalue
    return image

def robustmask(weight, factor=1, threshold=None):
    if threshold is None:
        w = sample(weight)
        q05, q15, q8, q99 = np.nanquantile(w, [0.05, 0.15, 0.8, 0.99])
        high_intensity = np.nanmean(w[(q8 <= w) & (w <= q99)])
        noise = np.nanmean(w[w <= q15])
        if noise > high_intensity / 10:
            noise = np.nanmean(w[w <= q05])
            if noise > high_intensity / 10:
                noise = 0  # no noise detected
        threshold = max(5 * noise, high_intensity / 5)
    
    mask = weight > (threshold * factor)
    
    # remove small holes and minimally grow
    boxsizes = [[5]] * weight.ndim
    mask = gaussiansmooth3d(mask, nbox=1, boxsizes=boxsizes) > 0.4
    mask = fill_holes(mask)
    boxsizes = [[3, 3]] * weight.ndim
    mask = gaussiansmooth3d(mask, nbox=2, boxsizes=boxsizes) > 0.6
    return mask

def mask_from_voxelquality(qmap, threshold='auto'):
    return robustmask(qmap)

def fill_holes(mask, max_hole_size=None):
    if max_hole_size is None:
        max_hole_size = mask.size / 20
    return ~morphology.remove_small_holes(~mask, area_threshold=max_hole_size)

def get_largest_connected_region(mask):
    labels = measure.label(mask)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return labels == largest_label

def brain_mask(mask, strength=7):
    # set border to false
    shrink_mask = mask.copy()
    if shrink_mask.ndim == 3 and all(s > 5 for s in shrink_mask.shape):
        shrink_mask[:,:,[0,-1]] = False
        shrink_mask[[0,-1],:,:] = False
        shrink_mask[:,[0,-1],:] = False
    
    boxsizes = [[strength]] * shrink_mask.ndim
    smoothed = gaussiansmooth3d(shrink_mask, nbox=1, boxsizes=boxsizes)
    shrink_mask2 = smoothed > 0.7
    
    brain_mask = get_largest_connected_region(shrink_mask2)
    
    # grow brain mask
    boxsizes = [[strength, strength]] * shrink_mask2.ndim
    smoothed = gaussiansmooth3d(brain_mask, nbox=2, boxsizes=boxsizes)
    brain_mask = smoothed > 0.2
    return brain_mask & mask

def sample(arr, n=10000):
    return np.random.choice(arr.flatten(), min(n, arr.size), replace=False)

#TODO Update function 
def gaussiansmooth3d(image, nbox=1, boxsizes=None):
    return ndimage.gaussian_filter(image, sigma=1)