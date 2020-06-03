from com.understandingclouds.constants import TRAIN_IMAGES_FOLDER, DATA_DIR
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from com.understandingclouds.plotting import get_image_masks
from com.understandingclouds.rle import masks_to_rle

train_images_files = [file for file in os.listdir(TRAIN_IMAGES_FOLDER) if os.path.isfile(os.path.join(
    TRAIN_IMAGES_FOLDER, file))]
train_df = pd.read_csv(DATA_DIR + 'train.csv')
train_df = train_df[['Image_Label', 'EncodedPixels']]
train_resized_df = pd.DataFrame(columns=train_df.columns)
aug_folder = 'images_augmented/'
if not os.path.isdir(aug_folder):
    os.mkdir(aug_folder)

for file in train_images_files:
    img = plt.imread(os.path.join(TRAIN_IMAGES_FOLDER, file))
    masks = get_image_masks(file, TRAIN_IMAGES_FOLDER, train_df, mask_height=320, mask_width=480, rle_height=1400,
                            rle_width=2100)

    resized_img = cv2.resize(img, (320, 480))

    train_resized_df = train_resized_df.append(masks_to_rle(file, masks))
    plt.imsave(os.path.join(aug_folder, file), resized_img)

train_resized_df.to_csv('train_resized.csv')
