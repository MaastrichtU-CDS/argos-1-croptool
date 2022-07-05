import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import time
import datetime
import random
import nibabel as nib
from shutil import copyfile
from tensorflow.keras import losses

import models
import utils
import data_augmentation


def run_once(f):
    """
    Wrapper for functions that should only run once every run.

    Parameters
    ----------
    f : function
        Function to be ran.

    Returns
    -------
    wrapper : boolean
    """

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice_score = self.add_weight(name='dsc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # smooth = 0.000001
        smooth = 1
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        score = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
        self.dice_score.assign(score)

    def result(self):
        return self.dice_score

    def reset_states(self):
        self.dice_score.assign(0.0)


def dice_loss(y_true, y_pred):
    smooth = 1
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    score = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return 1 - score


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """Soft dice loss calculation for arbitrary batch size, number of classes,
    and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch



def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = tf.ones(tf.shape(y_true))


    #ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones - y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = tf.math.reduce_sum(p0 * g0, axis=(0, 1, 2, 3))
    den = num + alpha * tf.math.reduce_sum(p0 * g1, axis=(0, 1, 2, 3)) + beta * tf.math.reduce_sum(p1 * g0, axis=(0, 1, 2, 3))

    # num = K.sum(p0*g0, (0,1,2,3))
    # den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    T = tf.math.reduce_sum(num/den)
    Ncl = tf.cast(tf.shape(y_true)[-1], dtype='float32')
    # T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    # Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def dice_score(y_true, y_pred, ignore_background=True, square=False):
    if ignore_background:
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
    y_pred_t = tf.where(tf.greater(y_pred, 0.15), 0, 1)
    y_pred_t = tf.dtypes.cast(y_pred_t, tf.float32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
    axes = (0, 1, 2)
    eps = 1e-7
    num = (2 * tf.reduce_sum(y_true * y_pred, axis=axes) + eps)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) + eps
    score = tf.reduce_mean(num / denom)

    return score


def dice_loss2(y_true, y_pred, ignore_background=False, square=False):
    if ignore_background:
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
    y_pred_t = tf.where(tf.greater(y_pred, 0.15), 0, 1)
    y_pred_t = tf.dtypes.cast(y_pred_t, tf.float32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
    axes = (0, 1, 2)
    eps = 1e-7
    num = (2 * tf.reduce_sum(y_true * y_pred, axis=axes) + eps)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) + eps
    score = tf.reduce_mean(num / denom)
    return 1 - score


def bce(y_true, y_pred):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return binary_cross_entropy(y_true, y_pred)


def dice_bce(y_true, y_pred):
    d_l = dice_loss2(y_true, y_pred)
    bce_l = bce(y_true, y_pred)
    return d_l + bce_l


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
        numbering = []
        contents = os.listdir(ct_path)
        for i, _ in enumerate(contents):
            numbering.append(i)
        # for layer, content in enumerate(numbering):
        for layer, content in enumerate(numbering):
            ct_fname = str(content) + '.nii.gz'
            gt_fname = str(content) + '_gtv.nii.gz'
            gt_patch_gtv = nib.load(os.path.join(gt_gtv_path, gt_fname)).get_fdata()

            if np.max(gt_patch_gtv) == 1:
                gt_slices.append(os.path.join(ct_path, ct_fname) + ',' + os.path.join(gt_gtv_path, gt_fname) + ',' + os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz') + ', ' + '1')

            else:
                gt_slices.append(os.path.join(ct_path, ct_fname) + ',' + os.path.join(gt_gtv_path, gt_fname) + ',' + os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz') + ', ' + '0')

        pos_dict[patient] = gt_pos
        neg_dict[patient] = gt_neg
        slice_dict[patient] = gt_slices

        with open(name, 'w') as fp:
            json.dump(slice_dict, fp)


def early_stopping(loss_list, min_delta=0.005, patience=20):
    """

    Parameters
    ----------
    loss_list : list
        List containing loss values for every evaluation.
    min_delta : float
        Float serving as minimum difference between loss values before early stopping is considered.
    patience : int
        Training will not be stopped before int(patience) number of evaluations have taken place.

    Returns
    -------

    """
    # TODO: Changed to list(loss_list)
    if len(list(loss_list)) // patience < 2:
        return False

    mean_previous = np.mean(loss_list[::-1][patience:2 * patience])
    mean_recent = np.mean(loss_list[::-1][:patience])
    delta_abs = np.abs(mean_recent - mean_previous)  # abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change

    if delta_abs < min_delta:
        print('Stopping early...')
        return True
    else:
        return False


@run_once
def _start_graph_tensorflow():
    """
    Starts the tensorboard graph. Allows for the tracking of loss curves, accuracy and architecture visualization.
    """
    tf.summary.trace_on(graph=True, profiler=True)


@run_once
def _end_graph_tensorflow(self, log_dir):
    """

    Parameters
    ----------
    self : tf.writer
        train_summary_writer.
    log_dir : str
        Path to directory where updates should be stored.

    Returns
    -------

    """
    with self.as_default():
        tf.summary.trace_export(name="graph", step=0, profiler_outdir=log_dir)


def get_sample(ct_path, gt_path, layer, params, augment=True):
    ct = np.zeros(shape=[params.dict['batch_size'], 512, 512, params.dict['patch_shape'][2]])
    gt = np.zeros(shape=[params.dict['batch_size'], 512, 512, 1])

    for batch_nr in range(0, params.dict['batch_size']):
        ct_patch = np.zeros([params.dict['patch_shape'][0],
                             params.dict['patch_shape'][1],
                             params.dict['patch_shape'][2]])
        min_layer = layer - params.dict['patch_shape'][2] // 2
        for z in range(0, params.dict['patch_shape'][-1]):
            ct_patch[:, :, z] = nib.load(os.path.join(ct_path, str(min_layer + z) + '.nii.gz')).get_fdata()
        gt_patch = nib.load(os.path.join(gt_path, str(layer) + '_gtv.nii.gz')).get_fdata()

        if augment:
            if random.randint(0, 1) == 1:
                num_augments = np.random.randint(1, params.dict['number_of_augmentations'] + 1)
                ct_patch, gt_patch = data_augmentation.apply_augmentations(ct_patch,
                                                                        gt_patch,
                                                                        num_augments)

        ct[batch_nr, :, :, :] = ct_patch
        gt[batch_nr, :, :, 0] = gt_patch
    gt = tf.one_hot(np.uint8(np.squeeze(gt, axis=-1)), params.dict['num_classes'])
    return ct, gt


def get_batch_full(ct_slices, params):
    ct = np.zeros(shape=[params.dict['batch_size'], 512, 512, params.dict['patch_shape'][2]])
    gt = np.zeros(shape=[params.dict['batch_size'], 512, 512, 1])

    for layer in range(0, params.dict['batch_size']):
        while True:
            random_case = random.choice(list(ct_slices))
            if len(ct_slices[random_case]) != 0:
                break
            else:
                print(str(random_case) + ' Length: ' + str(len(ct_slices[random_case])))

        rand_num = random.randint(0, 2)
        if rand_num == 0:
            while True:
                random_layer = random.randint(0, len(ct_slices[random_case]) - 1 - (params.dict['patch_shape'][2] // 2))
                selected_slice = ct_slices[random_case][random_layer]
                output = selected_slice.split(',')
                if int(output[-1]) == 1:

                    break
        else:
            random_layer = random.randint(0, len(ct_slices[random_case]) - 1 - (params.dict['patch_shape'][2] // 2))
            selected_slice = ct_slices[random_case][random_layer]
            output = selected_slice.split(',')

        min_layer = random_layer - params.dict['patch_shape'][2] // 2

        gt_patch = nib.load(output[1]).get_fdata()

        ct_patch = np.zeros([params.dict['patch_shape'][0],
                             params.dict['patch_shape'][1],
                             params.dict['patch_shape'][2]])

        for z in range(0, params.dict['patch_shape'][-1]):
            selected_slice = ct_slices[random_case][min_layer + z]
            output = selected_slice.split(',')
            ct_patch[:, :, z] = nib.load(output[0]).get_fdata()

        if random.randint(0, 1) == 1:
            num_augments = np.random.randint(1, params.dict['number_of_augmentations'] + 1)
            ct_patch, gt_patch = data_augmentation.apply_augmentations(ct_patch,
                                                                       gt_patch,
                                                                       num_augments)

        ct[layer, :, :, :] = ct_patch
        gt[layer, :, :, 0] = gt_patch
    gt = tf.one_hot(np.uint8(np.squeeze(gt, axis=-1)), params.dict['num_classes'])
    return ct, gt


def main():
    @tf.function
    def train_on_batch(im_src, gt_src):
        """
        Manages and updates parameters for training.
        Parameters
        ----------
        im_src : np.ndarray
        gt_src : np.ndarray
        pet_src : np.ndarray

        Returns
        -------

        """
        with tf.GradientTape() as tape:
            predictions = model(inputs=[im_src], training=True)
            regularization_loss = tf.math.add_n(model.losses)
            loss_value = loss_function(gt_src, predictions)
            total_loss = regularization_loss + loss_value

        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer_function.apply_gradients(zip(grads, model.trainable_weights))
        train_loss(total_loss)
        return predictions

    @tf.function
    def validate_on_batch(im_src, gt_src):
        """
        Manages validation.

        Parameters
        ----------
        im_src : np.ndarray
        gt_src : np.ndarray
        pet_src : np.ndarray

        Returns
        -------

        """
        predictions = model(inputs=[im_src], training=False)
        regularization_loss = tf.math.add_n(model.losses)
        loss_value = loss_function(gt_src, predictions)
        total_loss = regularization_loss + loss_value
        validation_loss(total_loss)
        return predictions

    param_path = os.getcwd() + '/params.json'
    params = utils.Params(param_path)

    sort_slices('/home/leroy/app/data/Train/',
                'slices_training_800200.json')

    sort_slices('/home/leroy/app/data/Validation/',
                'slices_validation_800200.json')

    # Define loss function
    loss_list = []
    # loss_function = losses.CategoricalCrossentropy()
    # loss_function = dice_loss2
    loss_function = dice_bce

    # Define optimizer with learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params.dict['learning_rate'],
                                                                  decay_steps=params.dict['decay_steps'],
                                                                  decay_rate=params.dict['decay_rate'],
                                                                  staircase=True)
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer_function = tf.keras.optimizers.Adam(params.dict['learning_rate'])

    # Define model
    model = models.mod_resnet(params,
                        params.dict['num_classes'],
                        optimizer=optimizer_function,
                        loss=loss_function)

    # print(model.summary)
    # Define evaluation metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = dice_score
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    validation_accuracy = dice_score

    # Create variables for various paths used for storing training information
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(params.dict['log_path']):
        os.mkdir(params.dict['log_path'])

    train_log_dir = params.dict['log_path'] + '/gradient_tape/' + current_time + '/train'
    val_log_dir = params.dict['log_path'] + '/gradient_tape/' + current_time + '/val'
    saved_model_path = params.dict['log_path'] + '/gradient_tape/' + current_time + '/saved_models/'
    saved_weights_path = params.dict['log_path'] + '/gradient_tape/' + current_time + '/saved_weights/'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    os.mkdir(saved_model_path)
    os.mkdir(saved_weights_path)

    # Load training and validation data
    train_slices = utils.read_slices('slices_training_800200.json')
    validation_slices = utils.read_slices('slices_validation_800200.json')

    patients_train = os.listdir('/home/leroy/app/data/Train/')
    patients_validation = os.listdir('/home/leroy/app/data/Validation/')
    # Start training loop
    epoch_number = []
    patient_id = []
    train_dice_scores = []
    train_loss_scores = []

    epoch_number_val = []
    patient_val_id = []
    val_dice_scores = []
    val_loss_scores = []
    for iteration in range(0, 1):
        # print(iteration)
        _start_graph_tensorflow()
        for patient in patients_train:
            print(f'Training on: {patient}')
            ct_path = os.path.join('/home/leroy/app/data/Train',
                                   patient + '/CT')
            gt_path = os.path.join('/home/leroy/app/data/Train',
                                   patient + '/GT/GTV')
            patient_contents = os.listdir(ct_path)
            for slice in range(1, len(patient_contents) - 1):
                ct_batch, gt_batch = get_sample(ct_path, gt_path, slice, params)
                train_pred = train_on_batch(ct_batch, gt_batch)

                # Evaluation step during training.
                # Write training information to training log
                with train_summary_writer.as_default():
                    train_dice = train_accuracy(gt_batch, train_pred)
                    tf.summary.scalar('loss', train_loss.result(), step=iteration)
                    tf.summary.scalar('accuracy', train_dice, step=iteration)
                template = 'Patient {}, Slice {}, Loss: {:.5}, Dice: {:.5}'
                print(template.format(patient,
                                    slice,
                                    train_loss.result(),
                                    train_dice))
                epoch_number.append(iteration)
                patient_id.append(patient)
                train_dice_scores.append(train_dice.numpy())
                train_loss_scores.append(train_loss.result().numpy())
        _end_graph_tensorflow(train_summary_writer, train_log_dir)

        for patient_val in patients_validation:
            print(f'Validating on: {patient_val}')
            ct_path = os.path.join('/home/leroy/app/data/Validation',
                                   patient_val + '/CT')
            gt_path = os.path.join('/home/leroy/app/data/Validation',
                                   patient_val + '/GT/GTV')
            patient_contents_val = os.listdir(ct_path)
            for slice in range(1, len(patient_contents_val) - 1):
                ct_batch_val, gt_batch_val = get_sample(ct_path,
                                                        gt_path,
                                                        slice,
                                                        params,
                                                        augment=False)
                val_pred = validate_on_batch(ct_batch_val, gt_batch_val)

                # Evaluation step during validation.
                # Write validation information to log
                with val_summary_writer.as_default():
                    # validation_dice = validation_accuracy(gt_batch_val, val_pred).numpy()
                    validation_dice = validation_accuracy(gt_batch_val, val_pred)
                    tf.summary.scalar('loss', validation_loss.result(), step=iteration)
                    tf.summary.scalar('accuracy', validation_dice, step=iteration)
                    loss_list.append(validation_loss.result())
                template = 'Patient {}, Slice {}, Validation Loss: {:.5}, Validation Dice: {:.5}'
                print(template.format(patient_val,
                                      slice,
                                      validation_loss.result(),
                                      validation_dice))
                epoch_number_val.append(iteration)
                patient_val_id.append(patient_val)
                val_dice_scores.append(validation_dice.numpy())
                val_loss_scores.append(validation_loss.result().numpy())
        # Save the model at predefined step numbers.
        # Hardcoded to save model every epoch
        if iteration % 1 == 0:
                model.save(os.path.join(saved_model_path,
                                        'model_' + str(iteration)))
                model.save_weights(os.path.join(saved_weights_path,
                                                'model_weights' + str(iteration) + '.h5'))
    df_train = pd.DataFrame({'epoch': epoch_number,
                             'patient_id': patient_id,
                             'train_dice': train_dice_scores,
                             'train_loss': train_loss_scores})
    df_val = pd.DataFrame({'epoch': epoch_number_val,
                           'patient_id': patient_val_id,
                           'val_dice': val_dice_scores,
                           'val_loss': val_loss_scores})
    df_train.to_csv(os.path.join(r'/home/leroy/app/data', 'training_logs.csv'), index=False)
    df_val.to_csv(os.path.join(r'/home/leroy/app/data', 'validation_logs.csv'), index=False)


if __name__ == '__main__':
    # Small check for GPU usage or CPU usage. CUDA_VISIBLE_DEVICES selects a
    # specific GPU card. Usefull when multiple people are training on the
    # same server.
    # CUDA_VISIBLE_DEVICES = 0
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    else:
        print("Running on CPU. Please install GPU version of TF")
    current_time = time.time()
    main()
    print(time.time() - current_time)
