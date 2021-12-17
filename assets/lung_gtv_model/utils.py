import numpy as np
import os
import nibabel as nib
import nrrd
import json
import random
import cv2
import copy
from skimage.morphology import binary_closing
from skimage.measure import label


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """"
        Save dict to json file

        Parameters
        ----------
        json_path : string
            Path to save location
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Load parameters from json file

        Parameters
        ----------
        json_path : string
            Path to json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """"
        Give dict-like access to Params instance by: 'params.dict['learning_rate']'
        """
        return self.__dict__


def load_nifti_set(patient_path):
    ct = nib.load(os.path.join(patient_path, 'image.nii.gz')).get_fdata()
    gt_lung1 = nib.load(os.path.join(patient_path, 'mask_Lung-Left.nii.gz')).get_fdata()
    gt_lung2 = nib.load(os.path.join(patient_path, 'mask_Lung-Right.nii.gz')).get_fdata()
    folder_contents = os.listdir(patient_path)
    gt_gtv = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'gtv' in s:
            gt_gtv += nib.load(os.path.join(patient_path, item)).get_fdata()
    # gt_gtv = nrrd.read(r'C:\Users\leroy.volmer\PycharmProjects\LungSegmentation\Data\lung001\GTV-1.nrrd')[0]
    gt_lung = gt_lung1 + gt_lung2
    return ct, gt_lung, gt_gtv



def read_slices(slice_json_fname):
    with open(slice_json_fname) as f:
        slices = json.load(f)

    return slices


def load_gtv_set(patient_path):
    folder_contents = os.listdir(patient_path)
    for item in folder_contents:
        s = item.lower()
        if 'image' in s:
            ct = nrrd.read(os.path.join(patient_path, item))[0]
    
    gt = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'gtv' in s:
            gt += nrrd.read(os.path.join(patient_path, item))[0]
    gt[gt != 0] = 1
    if np.max(gt) == 0:
        print(patient_path + ' Patient does not have a GTV in the GT.')
        
    gt_lung = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'lung-left' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lung-right' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lung-total' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lungs-total' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]

    gt_lung[gt_lung != 0] = 1
    return ct, gt, gt_lung


def load_gtv1_set(patient_path):
    folder_contents = os.listdir(patient_path)
    for item in folder_contents:
        s = item.lower()
        if 'image' in s:
            ct = nrrd.read(os.path.join(patient_path, item))[0]
    
    gt = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'gtv-1' in s:
            gt += nrrd.read(os.path.join(patient_path, item))[0]
    gt[gt != 0] = 1
    if np.max(gt) == 0:
        print(patient_path + ' Patient does not have a GTV in the GT.')
        
    gt_lung = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'lung-left' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lung-right' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lung-total' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lungs-total' in s:
            gt_lung += nrrd.read(os.path.join(patient_path, item))[0]
    gt_lung[gt_lung != 0] = 1
    return ct, gt, gt_lung

def load_nrrd_set(patient_path):
    folder_contents = os.listdir(patient_path)
    if '.DS_Store' in folder_contents:
        folder_contents.remove('.DS_Store')
    for item in folder_contents:
        s = item.lower()
        if 'image' in s:
            ct = nrrd.read(os.path.join(patient_path, item))[0]
    
    gt = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'gtv' in s:
            gt += nrrd.read(os.path.join(patient_path, item))[0]
    return ct, gt


def load_nrrd_lung_set(patient_path):
    folder_contents = os.listdir(patient_path)
    if '.DS_Store' in folder_contents:
        folder_contents.remove('.DS_Store')
    for item in folder_contents:
        s = item.lower()
        if 'image' in s:
            ct = nrrd.read(os.path.join(patient_path, item))[0]
    
    gt = np.zeros(shape=ct.shape)
    for item in folder_contents:
        s = item.lower()
        if 'lung-left' in s:
            gt += nrrd.read(os.path.join(patient_path, item))[0]
        if 'lung-right' in s:
            gt += nrrd.read(os.path.join(patient_path, item))[0]
    return ct, gt


def extract_bounding_boxes(gt_src):
    """
    Extract corner coordinates of bounding boxes from tumour locations. Finds contours of tumours and draws rectangles
    around these contours.

    Parameters
    ----------
    gt_src : ndarray
        Binary image containing tumour annotations

    Returns
    -------
    indices : list
        List containing corner coordinates of bounding boxes
    """
    im = gt_src * 255
    indices = []
    for layer in range(0, im.shape[2]):
        img = im[:, :, layer]
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx = 0
        for cnt in contours:
            idx += 1
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
            # roi=im[y:y+h,x:x+w]
            indices.append([x, x+w, y, y+h, layer])
    return indices


def load_samples(sample_txt_file, seed=42):
    """
    Load samples from a .txt file, extracts relevant information, splits between CT and GT, and shuffles samples
    according to a specified seed.
    Parameters
    ----------
    sample_txt_file : str
        Text file containing samples.
    seed : int - default = 42
        Seed for shuffling

    Returns
    -------
    samples_dict : dict
        Dictionary containing strings with locations to CT and GT patches.
    """

    with open(sample_txt_file, 'r') as infile:
        data = infile.readlines()

        ct_patches = []
        gt_patches = []
        for i in data:
            line = i.strip(',')
            line = line.split(',')

            ct_patches.append(line[0])
            gt_patches.append(line[1])
    array = list(zip(ct_patches, gt_patches))
    random.seed(seed)
    random.shuffle(array)
    ct_patches, gt_patches = zip(*array)
    samples_dict = {'ct_patches': list(ct_patches),
                    'gt_patches': list(gt_patches)
                    }

    return samples_dict


def shuffle_samples(samples, seed=42):
    """

    Parameters
    ----------
    samples : list
        List containing paths to patches
    seed : int
        Seed for shuffling

    Returns
    -------
    Shuffled Samples
    """
    random.seed(seed)
    return random.shuffle(list(samples))


def get_largest_cc(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return np.int8(largest_cc)


def load_batch(samples_dict, patch_path, iteration, batch_size):

    def _load_batch(sample_list, patch_path_dir):
        batch = []
        for sample_path in sample_list:
            patch = nib.load(os.path.join(patch_path_dir, sample_path)).get_fdata()
            # patch = np.expand_dims(patch, 0)
            # patch = np.expand_dims(patch, -1)
            batch.append(patch)
        # batch = np.expand_dims(batch, 0)
        return np.array(batch)

    min_index = (iteration * batch_size) - batch_size
    max_index = iteration * batch_size
    ct_samples = samples_dict['ct_patches'][min_index:max_index]
    gt_samples = samples_dict['gt_patches'][min_index:max_index]

    ct_batch = _load_batch(ct_samples, patch_path)
    gt_batch = _load_batch(gt_samples, patch_path)

    return ct_batch, gt_batch


def normalize(img, bound, min_bound, max_bound):
    """
    Normalize an image between "min_bound" and "max_bound", and scale between 0 and 1. If "bound" = 'True', scale
    between 2.5th and 97.5th percentile.
    Parameters
    ----------
    img : np.ndarray
        Image to normalize.
    bound : str - True or False.
        Whether to scale between percentiles.
    min_bound : int
        Lower bound for normalization.
    max_bound : int
        Upper bound for normalization.

    Returns
    -------
    img : np.ndarray
        Normalized and scaled image.
    """
    norm = 2.5
    img = (img - min_bound) / (max_bound - min_bound)
    img[img > 1] = 1
    img[img < 0] = 0
    if bound == 'True':
        mn = np.percentile(img, norm)
        mx = np.percentile(img, 100 - norm)
        a = (img - mn)
        b = (mx - mn)
        img = np.divide(a, b, np.zeros_like(a), where=b != 0)

    # print(np.min(img))
    # print(np.max(img))
    c = (img - np.min(img))
    d = (np.max(img) - np.min(img))
    img = np.divide(c, d, np.zeros_like(c), where=d != 0)

   
    # img += np.abs(img.min())
    # img *= 1/img.max()
    return img


# def normalize(img, bound, min_bound, max_bound):
#     """
#     Normalize an image between "min_bound" and "max_bound", and scale between 0 and 1. If "bound" = 'True', scale
#     between 2.5th and 97.5th percentile.
#     Parameters
#     ----------
#     img : np.ndarray
#         Image to normalize.
#     bound : str - True or False.
#         Whether to scale between percentiles.
#     min_bound : int
#         Lower bound for normalization.
#     max_bound : int
#         Upper bound for normalization.

#     Returns
#     -------
#     img : np.ndarray
#         Normalized and scaled image.
#     """
#     norm = 2.5
#     img = (img - min_bound) / (max_bound - min_bound)
#     img[img > 1] = 0
#     img[img < 0] = 0
#     if bound == 'True':
#         mn = np.percentile(img, norm)
#         mx = np.percentile(img, 100 - norm)
#         img = (img - mn) / (mx - mn)
   
#     img += np.abs(img.min())
#     img *= 1/img.max()
#     return img



def largest_component_mask(bin_img):
    """Finds the largest component in a binary image and returns the component as a mask."""

    contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # should be [1] if OpenCV 3+

    max_area = 0
    max_contour_index = 0
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)['m00']
        if contour_area > max_area:
            max_area = contour_area
            max_contour_index = i

    labeled_img = np.zeros(bin_img.shape, dtype=np.uint8)
    cv2.drawContours(labeled_img, contours, max_contour_index, color=255, thickness=-1)

    return labeled_img


def detach_table(ct_src):
    """
    Removes CT tables, couches and lead blankets from CT images. First determines an optimal threshold via Otsu's
    method to binarize the input image. Next filters horizontal and vertical lines.
    Parameters
    ----------
    ct_src : np.ndarray
        CT image to clean.
    Returns
    -------
    im : np.ndarray
        # TODO
    """
    
    ct_original = copy.deepcopy(ct_src)
    ct_original *= 255.0
    im = np.zeros(shape=ct_original.shape)
    for layer in range(0, np.shape(ct_original)[2]):
        a = ct_original[:, :, layer]
        img = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(gray, [c], -1, (255, 255, 255), 2)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(gray, [c], -1, (255, 255, 255), 2)
        
        # TODO Check wether invert gray or thresh
        a2 = np.invert(gray)
        # a2 = np.invert(thresh)
        a2[a2 != 0] = 255
        a2 = a2 / 255
        a2 = 1-a2
        a6 = binary_closing(a2, selem=np.ones((5, 5)))
        a7 = 1-a6
        im_layer = ct_original[:, :, layer] * a7
        im[:, :, layer] = im_layer
    return im


def segment_patient(detached_ct, ct_src):
    ct_original = copy.deepcopy(ct_src)
    ct_original *= 255.0
    cc = np.zeros(shape=ct_original.shape)
    new_im = np.zeros(shape=ct_original.shape)
    for layer in range(0, ct_original.shape[2]):
    
        bin_img = cv2.inRange(detached_ct[:, :, layer], 50, 225)
        component = largest_component_mask(bin_img)
        cc[:, :, layer] = component
        new_im[:, :, layer] = component * ct_original[:, :, layer]
    new_im = normalize_min_max(new_im).astype(np.float32)
    cc = cc/255
    return new_im, cc.astype(np.int8)


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
