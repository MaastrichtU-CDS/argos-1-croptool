import os
import json
import nibabel as nib
import numpy as np


def sort_slices(path, name):
    pos_dict = {}
    neg_dict = {}
    slice_dict = {}
    patients = os.listdir(path)
    for patient in patients:
    # patient = patients[0]
        
        patient_path = os.path.join(path, patient)
        ct_path = os.path.join(patient_path, 'CT')
        gt_path = os.path.join(patient_path, 'GT')
        gt_lung_path = os.path.join(gt_path, 'Lung')
        gt_gtv_path = os.path.join(gt_path, 'GTV')
        
        gt_pos = []
        gt_neg = []
        gt_slices = []
        for layer in range(0, len(os.listdir(ct_path))):
            
            # ct_patch = nib.load(os.path.join(ct_path, str(layer) + '_ct.nii.gz')).get_fdata()
            gt_patch_gtv = nib.load(os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz')).get_fdata()
            # pet_patch = nib.load(os.path.join(pet_path, str(layer) + '_pet.nii.gz')).get_fdata()
            if np.max(gt_patch_gtv) == 1:
                gt_slices.append(os.path.join(ct_path, str(layer) + '.nii.gz') + ',' + os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz') + ',' + os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz') + ', ' + '1')

                    
            else:
                gt_slices.append(os.path.join(ct_path, str(layer) + '.nii.gz') + ',' + os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz') + ',' + os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz') + ', ' + '0')

        pos_dict[patient] = gt_pos
        neg_dict[patient] = gt_neg
        slice_dict[patient] = gt_slices

        with open(name, 'w') as fp:
            json.dump(slice_dict, fp)        


sort_slices('/home/leroy/app/data/Train/',
            'slices_training_800200.json')

sort_slices('/home/leroy/app/data/Validation/',
            'slices_validation_800200.json')