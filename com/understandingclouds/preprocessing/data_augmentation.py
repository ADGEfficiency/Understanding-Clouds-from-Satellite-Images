from albumentations import HorizontalFlip, VerticalFlip, Rotate
from com.understandingclouds.data_generator import DataGenerator
from com.understandingclouds.constants import LABELS, TRAIN_IMAGES_FOLDER, DATA_DIR
import matplotlib.pyplot as plt
from com.understandingclouds.plotting import get_image_masks
from com.understandingclouds.rle import masks_to_rle
import os
import pandas as pd

train_images_files = [f for f in os.listdir(TRAIN_IMAGES_FOLDER) if
                      os.path.isfile(os.path.join(TRAIN_IMAGES_FOLDER, f))]
#train_images_files = train_images_files[:10]
train_df = pd.read_csv(DATA_DIR + 'train.csv')
aug_train_df = pd.read_csv('train_resized.csv')
dg = DataGenerator(TRAIN_IMAGES_FOLDER, train_images_files, train_df, 480, 320, 480, 320, LABELS)
train_df = train_df[['Image_Label', 'EncodedPixels']]
aug_train_df = aug_train_df[['Image_Label', 'EncodedPixels']]

aug_names = ['horizontal_flip', 'vertical_flip', 'rotated_45']
augs = [HorizontalFlip(p=1), VerticalFlip(p=1), Rotate(limit=(45, 45), p=1)]
aug_folder = 'images_augmented/'
if not os.path.isdir(aug_folder):
    os.mkdir(aug_folder)
for file in train_images_files:
    for i, aug in enumerate(augs):
        img = plt.imread(os.path.join(TRAIN_IMAGES_FOLDER, file))
        masks = get_image_masks(file, TRAIN_IMAGES_FOLDER, train_df, mask_height=320, mask_width=480, rle_height=1400,
                                rle_width=2100)
        augmented = aug(image=img, mask=masks)
        augmented_img = augmented['image']
        augmented_masks = augmented['mask']
        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]
        aug_filename = ''.join([filename, '_', aug_names[i], filetype])
        plt.imsave(os.path.join(aug_folder, aug_filename), augmented_img)
        aug_train_df = aug_train_df.append(masks_to_rle(aug_filename, augmented_masks))
        #plot_image_given_masks(img, masks)
        #plot_image_given_masks(augmented_img, augmented_masks)


aug_train_df.to_csv('augmented_train_df.csv')
