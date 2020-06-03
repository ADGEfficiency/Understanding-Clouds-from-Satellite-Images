import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
import os
from com.understandingclouds.constants import LABELS

def rle2mask(height, width, encoded):
    """
    taken from https://github.com/pudae/kaggle-understanding-clouds
    """
    if isinstance(encoded, float):
        img = np.zeros((height, width), dtype=np.uint8)
        return img

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


def get_image_masks(filename, folder, segmentations_df, mask_height=1400, mask_width=2100, rle_height=1400,
                    rle_width=2100):
    # the segmentations are in run length format
    filepath = os.path.join(folder, filename)
    image = mpimg.imread(filepath)
    masks = np.zeros((mask_height, mask_width, len(LABELS)))
    for i in range(len(LABELS)):
        segment_df = segmentations_df[segmentations_df.Image_Label == filename + '_' + LABELS[i]]
        assert len(segment_df) == 1
        segment = segment_df.iloc[0].EncodedPixels
        mask = rle2mask(rle_height, rle_width, segment)
        if (mask.shape[0] != mask_height) or (mask.shape[1] != mask_width):
            mask = cv2.resize(mask, (mask_width, mask_height))
        masks[:, :, i] = mask
    return masks


def mask_to_rle(img):
    """
    taken from https://github.com/pudae/kaggle-understanding-clouds
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def masks_to_rle(filename, masks):
    seg_dict = {}
    for i in range(len(LABELS)):
        seg_dict[filename + '_' + LABELS[i]] = mask_to_rle(masks[:, :, i])
    return pd.DataFrame(seg_dict.items(), columns=['Image_Label', 'EncodedPixels'])
