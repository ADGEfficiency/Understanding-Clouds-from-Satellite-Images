from com.understandingclouds.rle import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from com.understandingclouds.constants import LABELS


def get_segmentations(filename, df):
    return df[df.Image_Label.str.contains(filename)]


def plot_image_with_rle(filename, folder, segmentations):
    # segmentations are in run length format
    filepath = os.path.join(folder, filename)
    img = mpimg.imread(filepath)
    fig, axs = plt.subplots(4, figsize=(10, 15))
    for i in range(4):
        axs[i].imshow(img)
        axs[i].set_title(LABELS[i], fontsize=10)
        segment = segmentations[segmentations.Image_Label == filename + '_' + LABELS[i]].EncodedPixels.values[0]
        mask = rle2mask(320, 480, segment)
        axs[i].imshow(mask, alpha=0.5, cmap='gray')
    fig.suptitle('Segmentations for ' + filename, fontsize=15, y=1.08)
    plt.tight_layout(h_pad=2)
    plt.show()


def plot_image_with_masks(filename, folder, segmentattions_df, mask_height=1400, mask_width=2100, rle_height=1400,
                          rle_width=2100):
    masks = get_image_masks(filename, folder, segmentattions_df, mask_height, mask_width, rle_height, rle_width)
    filepath = os.path.join(folder, filename)
    img = mpimg.imread(filepath)
    fig, axs = plt.subplots(4, figsize=(10, 15))
    for i in range(4):
        axs[i].imshow(img)
        axs[i].set_title(LABELS[i], fontsize=20)
        axs[i].imshow(masks[:, :, i], alpha=0.5, cmap='gray')
    fig.suptitle('Segmentations for ' + filename, fontsize=24, y=1.08)
    plt.tight_layout(h_pad=1)


def plot_image_given_masks(image, masks):
    fig, axs = plt.subplots(4, figsize=(10, 15))
    for i in range(4):
        axs[i].imshow(image)
        axs[i].set_title(LABELS[i], fontsize=10)
        axs[i].imshow(masks[:, :, i], alpha=0.5, cmap='gray')
    plt.tight_layout(h_pad=2)
    plt.show()
