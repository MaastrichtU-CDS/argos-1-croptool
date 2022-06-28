import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import label
from scipy import ndimage

import utils


def load_io1(path):

    ct_path = os.path.join(path, 'CT')
    gt_path = os.path.join(path, 'GT/GTV')

    ct = np.zeros([512, 512, len(os.listdir(ct_path))])

    gt = np.zeros([512, 512, len(os.listdir(ct_path))])
    for i, content in enumerate(os.listdir(ct_path)):
        slice = nib.load(os.path.join(ct_path, content)).get_fdata()
        ct[:, :, i] = slice
    for i, content in enumerate(os.listdir(gt_path)):
        gt_slice = nib.load(os.path.join(gt_path, content)).get_fdata()
        gt[:, :, i] = gt_slice


    return ct, gt


def calculate_pr_f1(gt, pred):
    testImage = sitk.GetImageFromArray(gt)
    testImage = sitk.Cast(testImage, sitk.sitkUInt32)
    resultImage = sitk.GetImageFromArray(pred)
    resultImage = sitk.Cast(resultImage, sitk.sitkUInt32)

    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1, nWMH, nDetections


def pad_img(img, pad_shape, params):
    img = np.pad(img, ((0, pad_shape),
                      (0, pad_shape),
                      (params.dict['patch_shape'][2] // 2, params.dict['patch_shape'][2] // 2)),
                  'symmetric')
    return img


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


def get_predictions():
    param_path = os.getcwd() + '/assets/lung_gtv_model/params.json'
    params = utils.Params(param_path)
    # Specify entire folder, not saved_model.pb
    loaded = tf.keras.models.load_model(os.getcwd() + '/assets/lung_gtv_model/saved_models/model_2000000', compile=False)

    patients_train = os.listdir('/home/leroy/app/data/Train/')
    patients_validation = os.listdir('/home/leroy/app/data/Validation/')

    patient_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    ndetections_list = []
    nlesions_list = []
    d1_list = []
    ct_shape = []
    gt_shape = []

    for patient in patients_train:
        print(patient)
        ct, gt = load_io1(os.path.join('/home/leroy/app/data/Train', patient))
        # print(f'Patient: {patient} has CT shape: {np.shape(ct)} and GT shape: {np.shape(gt)}')
        # pred_crop = pred_lung[:, :, min_layer_pred2:max_layer_pred2]

        predictions = pred_img(ct, loaded, params)

        dsc = (2 * np.sum(predictions * gt)) / (np.sum(predictions) + np.sum(gt))
        precision, recall, f1, nWMH, nDetections = calculate_pr_f1(gt, predictions)
        patient_list.append(patient)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        ndetections_list.append(nDetections)
        nlesions_list.append(nWMH)
        d1_list.append(dsc)
        ct_shape.append(np.shape(ct))
        gt_shape.append(np.shape(gt))
    data = {
        'Patient': patient_list,
        'CT_Shape': ct_shape,
        'GT_Shape': gt_shape,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1': f1_list,
        'nDetections': ndetections_list,
        'nLesions': nlesions_list,
        'Dice1': d1_list
    }
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(r'/home/leroy/app/data', 'train_results.csv'), index=False)

    # Quick and dirty copy. Ugly but works for testing purposes.
    patient_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    ndetections_list = []
    nlesions_list = []
    d1_list = []
    ct_shape = []
    gt_shape = []

    for patient in patients_validation:
        print(patient)
        ct, gt = load_io1(os.path.join('/home/leroy/app/data/Validation', patient))
        predictions = pred_img(ct, loaded, params)

        dsc = (2 * np.sum(predictions * gt)) / (np.sum(predictions) + np.sum(gt))
        precision, recall, f1, nWMH, nDetections = calculate_pr_f1(gt, predictions)
        patient_list.append(patient)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        ndetections_list.append(nDetections)
        nlesions_list.append(nWMH)
        d1_list.append(dsc)
        ct_shape.append(np.shape(ct))
        gt_shape.append(np.shape(gt))
    data = {
        'Patient': patient_list,
        'CT_Shape': ct_shape,
        'GT_Shape': gt_shape,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1': f1_list,
        'nDetections': ndetections_list,
        'nLesions': nlesions_list,
        'Dice1': d1_list
    }
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(r'/home/leroy/app/data', 'validation_results.csv'), index=False)



if __name__ == '__main__':
    # print(tf.test.gpu_device_name())
    print('Starting predictions')
    get_predictions()
