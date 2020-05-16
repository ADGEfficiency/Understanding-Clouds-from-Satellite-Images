from com.understandingclouds.data_generator import DataGenerator
import cv2
import numpy as np
import pandas as pd
from com.understandingclouds.rle import mask_to_rle
from com.understandingclouds.constants import LABELS


def predict_dataset(model, folder, file_list, resize=(320, 480), threshold=0.5):
    datagen = DataGenerator(folder, file_list, None, 480, 320, 480, 320, LABELS, to_fit=False, batch_size=32)
    preds = np.empty((len(file_list), resize[0], resize[1], 4), dtype=np.int8)
    for i in range(datagen.__len__()):
        print(i)
        pred = model.predict(datagen.__getitem__(i))
        if resize != (320, 480):
            pred_resized = [cv2.resize(pred[i, :, :, :], (resize[1], resize[0])) for i in range(pred.shape[0])]
            pred = np.asarray(pred_resized)
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        preds[i * 32:(i + 1) * 32, :, :, :] = pred.astype(np.int8)

    res_df = pd.DataFrame()
    fish_list = list(map(lambda x: x + '_Fish', file_list))
    flower_list = list(map(lambda x: x + '_Flower', file_list))
    gravel_list = list(map(lambda x: x + '_Gravel', file_list))
    sugar_list = list(map(lambda x: x + '_Sugar', file_list))

    all_list = ['all'] * len(file_list) * 4
    all_list[0::4] = fish_list
    all_list[1::4] = flower_list
    all_list[2::4] = gravel_list
    all_list[3::4] = sugar_list

    res_df['Image_Label'] = all_list

    masks_fish = preds[:, :, :, 0]
    masks_flower = preds[:, :, :, 1]
    masks_gravel = preds[:, :, :, 2]
    masks_sugar = preds[:, :, :, 3]

    all_masks = np.empty((len(file_list) * 4, resize[0], resize[1]), dtype=np.int8)
    all_masks[0::4] = masks_fish
    all_masks[1::4] = masks_flower
    all_masks[2::4] = masks_gravel
    all_masks[3::4] = masks_sugar

    rle_list = [mask_to_rle(all_masks[i]) for i in range(all_masks.shape[0])]
    res_df['EncodedPixels'] = rle_list
    return res_df, preds