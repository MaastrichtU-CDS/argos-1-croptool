import numpy as np
import random
import skimage
import skimage.transform
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage


def normalize_min_max(img):
    """
    Normalize data between 0 and 1.

    Parameters
    ----------
    img : numpy.ndarray
        image.

    Returns
    -------
    img
        Normalized image between 0 and 1.

    """
    a = (img - np.min(img))
    b = (np.max(img) - np.min(img))
    return np.divide(a, b, np.zeros_like(a), where=b != 0)


def vertical_flip(img):
    """
    Flip an image along the x-axis.

    Parameters
    ----------
    img : numpy.ndarray
        Image.

    Returns
    -------
    img
        Flipped image.

    """
    return np.flipud(np.copy(img))


def horizontal_flip(img):
    """
    Flips an input image horizontally (mirror)

    Parameters
    ----------
    img : numpy.ndarray
        The image to flip.

    Returns
    -------
    TYPE
        Flipped version of image.

    """
    return np.fliplr(np.copy(img))


def rot90_flip(img, k_times):
    """
    Rotate and image 90 degrees 'k_times'.

    Parameters
    ----------
    img : numpy.ndarray
        Image.

    Returns
    -------
    img
        Flipped image.

    """
    return np.rot90(np.copy(img), k=k_times)



def upside_down_flip(img):
    """
    Flip a 3D array upside down [:, :, 0] --> [:, :, -1].

    Parameters
    ----------
    img : numpy.ndarray
        The image to flip.

    Returns
    -------
    img
        Flipped version of img.

    """
    return np.copy(img)[:, :, ::-1]


def rotate(img, rad):
    """
    Rotates the source image rad degrees.

    Parameters
    ----------
    img : numpy.ndarray
        The image to rotate.
    rad : number
        degrees to rate img.

    Returns
    -------
    img2 : numpy.ndarray
        Rotated version of img.

    """
    img2 = np.copy(img)
    if np.ndim(img2) == 3:
        for z in range(0, img2.shape[-1]):
            img2[:, :, z] = skimage.transform.rotate(img2[:, :, z],
                                                     rad,
                                                     preserve_range=True)
    elif np.ndim(img2) == 2:
        img2 = skimage.transform.rotate(img2, rad, preserve_range=True)
    return img2


def blur_gaussian(img, sigma, eps=1e-3):
    """
    Blur an image using gaussian blurring.

    Parameters
    ----------
    img : numpy.ndarray
        The image to blur. Expected to be of shape ``(H, W)`` or ``(H, W, C)``.
    sigma : number
        Standard deviation of the gaussian blur. Larger numbers result in
        more large-scale blurring.
    eps : number, optional
        A threshold used to decide whether `sigma` can be considered zero.
        The default is 1e-3.

    Returns
    -------
    img : numpy.ndarray
        The blurred image with identical shape and datatype as the input img.

    """
    if sigma < eps:
        return img

    if img.ndim == 2:
        img[:, :] = ndimage.gaussian_filter(img[:, :],
                                            sigma,
                                            mode="mirror")
    else:
        nb_channels = img.shape[2]
        for channel in range(nb_channels):
            img[:, :, channel] = ndimage.gaussian_filter(img[:, :, channel],
                                                         sigma,
                                                         mode="mirror")
    return img


def gamma_correction(img, gamma, gain=1):
    """
    Perform pixel-wise gamma correction on the input image according to the
    Power Law transformation: output = ct_src ** gamma.

    Parameters
    ----------
    img : numpy.ndarray
        DESCRIPTION.
    gamma : float
        DESCRIPTION.
    gain : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    adjusted_gamma_image : numpy.ndarray
        DESCRIPTION.

    """
    if np.min(img) < 0:
        img = normalize_min_max(img)
    adjusted_gamma_image = exposure.adjust_gamma(img, gamma=gamma, gain=gain)
    return adjusted_gamma_image


def shear_transform(img, shearing_factor):
    """
    Shear angle in counter-clockwise direction as radians.

    Parameters
    ----------
    img : numpy.ndarray
        Image.
    shearing_factor : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shear_transformer = skimage.transform.AffineTransform(shear=shearing_factor)
    return skimage.transform.warp(img, inverse_map=shear_transformer.inverse, preserve_range=True)


def random_noise_augment(im):
    """
    

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    sigma = random.uniform(0, 0.0001)
    return random_noise(im, var=sigma)


def adaptive_hist_equalization(im):
    """
    

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.

    Returns
    -------
    im : TYPE
        DESCRIPTION.

    """
    im = exposure.equalize_adapthist(im, clip_limit=0.005, nbins=512)
    return im


# def invert_colours(ct_src):
#     ct = 1 - ct_src
#     return ct


def invert_colours_set(ct_src, gt_src):
    # ct = invert_colours(ct_src)
    ct = 1 - ct_src
    return ct, gt_src


def adaptive_hist_equalization_set(ct_src, gt_src):
    ct = adaptive_hist_equalization(ct_src)
    return ct, gt_src


def random_noise_set(ct_src, gt_src):
    ct = random_noise_augment(ct_src)
    return ct, gt_src 


def shear_transform_set(ct_src, gt_src):
    shearing_factor = random.uniform(-0.1, 0.1)
    ct = shear_transform(ct_src, shearing_factor)
    gt = shear_transform(gt_src, shearing_factor)
    return ct, gt.astype(np.int16)


def gamma_correction_set(ct_src, gt_src):
    gamma = random.uniform(0.8, 1.2)
    ct = gamma_correction(ct_src, gamma)
    return ct, gt_src


def blur_gaussian_set(ct_src, gt_src):
    sigma = np.random.uniform(0, 1.5)
    ct = blur_gaussian(ct_src, sigma)
    # gt = blur_gaussian(gt_src, sigma)
    return ct, gt_src


def rotate_set(ct_src, gt_src):
    rad = np.random.randint(-30, 30)
    ct = rotate(ct_src, rad)
    gt = rotate(gt_src, rad)
    return ct, gt.astype(np.int16)


def horizontal_flip_set(ct_src, gt_src):
    ct = horizontal_flip(ct_src)
    gt = horizontal_flip(gt_src)
    return ct, gt


def rot90_flip_set(ct_src, gt_src):
    x = np.random.randint(0, 4)
    ct = rot90_flip(ct_src, k_times=x)
    gt = rot90_flip(gt_src, k_times=x)
    return ct, gt

def vertical_flip_set(ct_src, gt_src):
    ct = vertical_flip(ct_src)
    gt = vertical_flip(gt_src)
    return ct, gt


def upside_down_flip_set(ct_src, gt_src):
    ct = upside_down_flip(ct_src)
    gt = upside_down_flip(gt_src)
    return ct, gt


def get_augmentations():
    """
    Pool augmentations into a single function

    Returns
    -------
    augmentations : list
        list with all possible augmentation functions.

    """
    augmentations = [horizontal_flip_set,
                     vertical_flip_set,
                     rotate_set,
                     blur_gaussian_set,
                     rot90_flip_set,
                     # invert_colours_set
                     gamma_correction_set,
                     shear_transform_set
                     # random_noise_set,
                     # adaptive_hist_equalization_set
                     ]
    return list(augmentations)


def apply_augmentations(ct_src, gt_src, num_augmentations):
    augmenters = get_augmentations()
    if random.randint(0, 1) == 1:
        augmentations = random.sample(list(augmenters), int(num_augmentations))
        for augmenter in augmentations:
            ct_patch, gt_patch = augmenter(ct_src,
                                           gt_src)
            ct_patch = normalize_min_max(ct_patch)
        return ct_patch, gt_patch
    else:
        return ct_src, gt_src
