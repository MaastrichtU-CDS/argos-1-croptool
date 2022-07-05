import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import nibabel as nib



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


def get_predictions():
    patients_train = os.listdir('/home/leroy/app/data/Train/')
    patients_validation = os.listdir('/home/leroy/app/data/Validation/')

    patient_list = []
    ct_shape = []
    gt_shape = []

    for patient in patients_train:
        print(patient)
        ct, gt = load_io1(os.path.join('/home/leroy/app/data/Train', patient))
        patient_list.append(patient)
        ct_shape.append(np.shape(ct))
        gt_shape.append(np.shape(gt))
    data = {
        'Patient': patient_list,
        'CT_Shape': ct_shape,
        'GT_Shape': gt_shape
    }
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(r'/home/leroy/app/data', 'train_shapes.csv'), index=False)

    # Quick and dirty copy. Ugly but works for testing purposes.
    patient_list = []

    ct_shape = []
    gt_shape = []

    for patient in patients_validation:
        print(patient)
        ct, gt = load_io1(os.path.join('/home/leroy/app/data/Validation', patient))
        patient_list.append(patient)
        ct_shape.append(np.shape(ct))
        gt_shape.append(np.shape(gt))
    data = {
        'Patient': patient_list,
        'CT_Shape': ct_shape,
        'GT_Shape': gt_shape

    }
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(r'/home/leroy/app/data', 'validation_shapes.csv'), index=False)



if __name__ == '__main__':
    # print(tf.test.gpu_device_name())
    print('Starting loading and shapes check')
    get_predictions()
