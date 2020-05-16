from keras.losses import  binary_crossentropy
import numpy as np
import keras.backend as K



def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.dot(y_true_f, y_pred_f)
    return (2 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_coef_tf(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def mean_dice_coef(datagen, preds):
    dice_coefs = np.empty(preds.shape[0] * 4, dtype = np.float32)
    for i in range(datagen.__len__()):
        curr_batch = datagen.__getitem__(i)[1]
        for j in range(len(curr_batch)):
            sample_ind = (i * len(datagen.__getitem__(0)[1])) + j
            for k in range(4):
                ind = (sample_ind * 4) + k
                dice_coefs[ind] = dice_coef(curr_batch[j, :, :, k], preds[sample_ind, :, :, k])
    return np.mean(dice_coefs)