import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='2.0'
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy import ndimage

import utils


def get_cc(pred, thresh):
    label_img, cc_num = label(pred)
    # CC = find_objects(label_img)
    cc_areas = ndimage.sum(pred, label_img, range(cc_num+1))
    area_mask = (cc_areas < thresh)
    label_img[area_mask[label_img]] = 0
    return label_img, cc_areas


def load_io1(path):
    contents = os.listdir(path)
    for content in contents:
        if 'image' in content.lower():
            ct = nib.load(os.path.join(path, content)).get_fdata()

    gt1 = np.zeros(shape=[ct.shape[0], ct.shape[1], ct.shape[2]]).astype(np.uint8)
    gt2 = np.zeros(shape=[ct.shape[0], ct.shape[1], ct.shape[2]]).astype(np.uint8)
    gt3 = np.zeros(shape=[ct.shape[0], ct.shape[1], ct.shape[2]]).astype(np.uint8)
    gt4 = np.zeros(shape=[ct.shape[0], ct.shape[1], ct.shape[2]]).astype(np.uint8)
    gt5 = np.zeros(shape=[ct.shape[0], ct.shape[1], ct.shape[2]]).astype(np.uint8)
    for content in contents:
        if '1vis-1' in content.lower():
            gt1 = nib.load(os.path.join(path, content)).get_fdata()
        if '1vis-2' in content.lower():
            gt2 = nib.load(os.path.join(path, content)).get_fdata()
        if '1vis-3' in content.lower():
            gt3 = nib.load(os.path.join(path, content)).get_fdata()
        if '1vis-4' in content.lower():
            gt4 = nib.load(os.path.join(path, content)).get_fdata()
        if '1vis-5' in content.lower():
            gt5 = nib.load(os.path.join(path, content)).get_fdata()

    gt1[gt1 > 0] = 1
    gt2[gt2 > 0] = 1
    gt3[gt3 > 0] = 1
    gt4[gt4 > 0] = 1
    gt5[gt5 > 0] = 1
    return ct, gt1, gt2, gt3, gt4, gt5


def pad_img(img, pad_shape, params):
    img = np.pad(img, ((0, pad_shape),
                      (0, pad_shape),
                      (params.dict['patch_shape'][2] // 2, params.dict['patch_shape'][2] // 2)),
                  'symmetric')
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



def pred_img(ct, loaded, params):

    new_shape = ct.shape[0]
    pad_shape = new_shape - ct.shape[0]

    ct2 = pad_img(ct, pad_shape, params)

    predictions = np.zeros([ct2.shape[0], ct2.shape[1], ct.shape[2], params.dict['num_classes']])

    for z in range(0, int(ct.shape[2])):  # / params.dict['patch_shape'][2]
        ct_layer = np.expand_dims(ct2[:, :, z:z + params.dict['patch_shape'][2]], 0)
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


def get_predictions(path):
    param_path = os.getcwd() + '/assets/lung_gtv_model/params.json'
    params = utils.Params(param_path)
    # Specify entire folder, not saved_model.pb
    loaded_l = tf.keras.models.load_model(os.getcwd() + '/assets/lung_volume_model/saved_models/model_23800', compile=False)
    loaded = tf.keras.models.load_model(os.getcwd() + '/assets/lung_gtv_model/saved_models/model_2000000', compile=False)

    for patient in os.listdir(path):
        print(patient)
        ct, gt1, gt2, gt3, gt4, gt5 = load_io1(os.path.join(path, patient))
        ct_norm = utils.normalize_min_max(ct)
        detached_ct = utils.detach_table(ct_norm)
        ct_norm, cc = utils.segment_patient(detached_ct, ct_norm)


        ct = utils.normalize(ct, 'False', params.dict['min_bound'], params.dict['max_bound'])
        pred_lung = pred_img_l(ct_norm, loaded_l, params)
        pred_eroded = ndimage.morphology.binary_erosion(pred_lung, structure=np.ones((2, 2, 2)))
        label_img, cc_areas = get_cc(pred_eroded, thresh=50000)
        preds2 = ndimage.morphology.binary_dilation(label_img, structure=np.ones((1, 1, 1)))
        preds2 = preds2.astype(np.uint8)

        pred2_sort = np.argwhere(preds2 == 1)
        pred2_sorted = pred2_sort[:, 2]
        min_layer_pred2 = np.min(pred2_sorted)
        max_layer_pred2 = np.max(pred2_sorted)

        tolerance = 2

        if min_layer_pred2 - tolerance < 0:
            min_layer_pred2 = 0
        if max_layer_pred2 + tolerance > np.shape(ct)[2]:
            max_layer_pred2 = np.shape(ct)[2]

            # TODO: Added ct_norm2 instead of ct
        ct_crop = ct[:, :, min_layer_pred2:max_layer_pred2]
        gt1 = gt1[:, :, min_layer_pred2:max_layer_pred2]
        gt2 = gt2[:, :, min_layer_pred2:max_layer_pred2]
        gt3 = gt3[:, :, min_layer_pred2:max_layer_pred2]
        gt4 = gt4[:, :, min_layer_pred2:max_layer_pred2]
        gt5 = gt5[:, :, min_layer_pred2:max_layer_pred2]
        # pred_crop = pred_lung[:, :, min_layer_pred2:max_layer_pred2]

        predictions = pred_img(ct_crop, loaded, params)

        overlay_1 = (2 * np.sum(predictions * gt1)) / (np.sum(predictions) + np.sum(gt1))
        overlay_2 = (2 * np.sum(predictions * gt2)) / (np.sum(predictions) + np.sum(gt2))
        overlay_3 = (2 * np.sum(predictions * gt3)) / (np.sum(predictions) + np.sum(gt3))
        overlay_4 = (2 * np.sum(predictions * gt4)) / (np.sum(predictions) + np.sum(gt4))
        overlay_5 = (2 * np.sum(predictions * gt5)) / (np.sum(predictions) + np.sum(gt5))

        print(overlay_1)
        print(overlay_2)
        print(overlay_3)
        print(overlay_4)
        print(overlay_5)
        print()


if __name__ == '__main__':
    # print(tf.test.gpu_device_name())
    print('Starting predictions')
    get_predictions(path=os.getcwd() + '/data/Test')
