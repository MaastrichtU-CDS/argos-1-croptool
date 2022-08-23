# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='2.0'
import numpy as np
import pandas as pd
import nrrd
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf
from scipy.ndimage import label
from scipy import ndimage
from scipy.ndimage import find_objects, sum as scipy_sum

import utils

def load_im(patient_path):
    ct_path = os.path.join(patient_path, 'CT')
    gt_path = os.path.join(patient_path, 'GT')

    if os.path.exists(ct_path):
        ct = nrrd.read(os.path.join(ct_path, 'ct_image.nrrd'))[0]

    else:
        ct = 0

    if os.path.exists(gt_path):
        gt = np.zeros(ct.shape)
        gt_contents = os.listdir(gt_path)
        for content in gt_contents:
            gt  += nrrd.read(os.path.join(gt_path, content))[0]
    else:
        gt = 0

    return ct, gt


def _save_nifti(img, path):
    img = nib.Nifti1Image(img, np.eye(4))
    img.header.get_xyzt_units()
    nib.save(img, path)



def get_cc(pred, thresh):
    label_img, cc_num = label(pred)
    # CC = find_objects(label_img)
    cc_areas = ndimage.sum(pred, label_img, range(cc_num+1))
    area_mask = (cc_areas < thresh)
    label_img[area_mask[label_img]] = 0
    return label_img, cc_areas


def pad_img(img, pad_shape, params):
    img = np.pad(img, ((0, pad_shape),
                       (0, pad_shape),
                       (params.dict['patch_shape'][2] // 2, params.dict['patch_shape'][2] // 2)),
                 'symmetric')
    return img


def crop_img(img, min_crop, max_crop):
    img[img < min_crop] = min_crop
    img[img > max_crop] = max_crop
    return img


def pred_img_l(ct, loaded, params):

    new_shape = ct.shape[0]
    pad_shape = new_shape - ct.shape[0]

    ct2 = pad_img(ct, pad_shape, params)

    predictions = np.zeros([ct2.shape[0], ct2.shape[1], ct.shape[2], params.dict['num_classes']])

    for z in range(0, int(ct.shape[2])):  # / params.dict['patch_shape'][2]
        ct_layer = np.expand_dims(ct2[:, :, z], 0)
        # ct_layer = np.expand_dims(ct2[:, :, z], -1)
        # TODO check this expand
        # ct_layer = np.expand_dims(ct_layer, -1)

        pred = loaded.predict([ct_layer])
        predictions[:, :, z, :] = pred[0, :, :, :]
        # print(z)

    predictions[predictions < 0.15] = 0
    predictions[predictions != 0] = 1
    predictions = predictions[:, :, :, 1]

    return predictions


def crop(data_src, save_path, params, fname):
    # Load Model
    loaded_l = tf.keras.models.load_model(os.getcwd() + '/assets/lung_volume_model/saved_models/model_23800', compile=False)

    # Load image set
    patients = os.listdir(data_src)

    patient_list = []
    patient_shape_list = []
    # Predict lung segmentation on image
    for patient in patients:
        print(patient)
        patient_path = os.path.join(data_src, patient)
        ct, gt_gtv = load_im(patient_path)

        if np.max(ct) == 0:
            continue
        if np.max(gt_gtv) == 0:
            continue

        if np.shape(ct) != np.shape(gt_gtv):
            print(patient + 'CT and GT shape mismatch, skipping...')
        elif np.max(gt_gtv) == 0:
            print(patient + 'GT is empty, skipping...')
        else:
            ct = crop_img(ct, -1024, 3071)
            ct_norm = utils.normalize_min_max(ct)
            # ct_norm = utils.normalize(ct, 'True', params.dict['min_bound'], params.dict['max_bound'])
            detached_ct = utils.detach_table(ct_norm)
            ct_norm, cc = utils.segment_patient(detached_ct, ct_norm)

            pred_lung = pred_img_l(ct_norm, loaded_l, params)

            pred_eroded = ndimage.morphology.binary_erosion(pred_lung, structure=np.ones((2, 2, 2)))
            label_img, cc_areas = get_cc(pred_eroded, thresh=50000)
            preds2 = ndimage.morphology.binary_dilation(label_img, structure=np.ones((1, 1, 1)))
            preds2 = preds2.astype(np.uint8)

            pred2_sort = np.argwhere(preds2 == 1)
            if pred2_sort.size == 0:
                print(f'Patient: {patient} failed to predict lung volume, skipping...')
                continue
            else:
                pred2_sorted = pred2_sort[:, 2]
                min_layer_pred2 = np.min(pred2_sorted)
                max_layer_pred2 = np.max(pred2_sorted)

                tolerance = 2

                if min_layer_pred2 - tolerance < 0:
                    min_layer_pred2 = 0
                if max_layer_pred2 + tolerance > np.shape(ct)[2]:
                    max_layer_pred2 = np.shape(ct)[2]

                ct_crop = ct[:, :, min_layer_pred2:max_layer_pred2]
                print(f'Patient: {patient} has {np.shape(ct_crop)} CT shape')
                gt_gtv_crop = gt_gtv[:, :, min_layer_pred2:max_layer_pred2]
                ct_crop = utils.normalize(ct_crop, 'False', params.dict['min_bound'], params.dict['max_bound'])
                patient_save_path = os.path.join(save_path, patient)
                ct_path = os.path.join(patient_save_path, 'CT')
                gt_path = os.path.join(patient_save_path, 'GT')
                gt_gtv_path = os.path.join(gt_path, 'GTV')

                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if not os.path.exists(patient_save_path):
                    os.mkdir(patient_save_path)
                if not os.path.exists(ct_path):
                    os.mkdir(ct_path)
                if not os.path.exists(gt_path):
                    os.mkdir(gt_path)
                if not os.path.exists(gt_gtv_path):
                    os.mkdir(gt_gtv_path)

                for layer in range(ct_crop.shape[2]):
                    _save_nifti(ct_crop[:, :, layer],
                                os.path.join(ct_path, str(layer) + '.nii.gz'))

                    _save_nifti(gt_gtv_crop[:, :, layer],
                                os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz'))

                patient_list.append(patient)
                patient_shape_list.append(np.shape(ct_crop))
    df = pd.DataFrame({'patient': patient_list,
                       'patient_shape': patient_shape_list})
    df.to_csv(os.path.join(r'/home/leroy/app/data', fname), index=False)


if __name__ == '__main__':
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Running on CPU. Please install GPU version of TF")

    param_path = os.getcwd() + '/assets/lung_gtv_model/params.json'
    params = utils.Params(param_path)
    save_path = os.getcwd() + '/data'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path_train = os.path.join(save_path, 'Train')
    save_path_validation = os.path.join(save_path, 'Validation')
    crop(r'/home/leroy/app/data/pre-process-TRAIN', save_path_train, params, 'train_list.csv')
    crop(r'/home/leroy/app/data/pre-process-VALIDATE', save_path_validation, params, 'validation_list.csv')
    print('Saving "train_list.csv" and "validation_list.csv" logs to data path')

# %%

