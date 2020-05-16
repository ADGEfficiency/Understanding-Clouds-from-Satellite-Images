from tensorflow.keras.utils import Sequence
import numpy as np
import os
import matplotlib.image as mpimg
import cv2
from com.understandingclouds.rle import get_image_masks


class DataGenerator(Sequence):

    def __init__(self, images_folder, images_list, segmentations_df, image_width,
                 image_height, rle_width, rle_height, labels, to_fit=True, batch_size=32):
        self.images_folder = images_folder
        self.images_list = images_list
        self.segmentations_df = segmentations_df
        self.labels = labels
        self.image_width = image_width
        self.image_height = image_height
        self.rle_width = rle_width
        self.rle_height = rle_height
        self.to_fit = to_fit
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.images_list) / self.batch_size))

    def __getitem__(self, index):
        curr_images = [self.images_list[i] for i in
                       range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.images_list)))]
        X = self._get_X(curr_images)

        if not self.to_fit:
            return X

        Y = self._get_Y(curr_images)

        return X, Y

    def _get_X(self, images):
        X = np.empty((len(images), self.image_height, self.image_width, 3), dtype=np.int16)
        for i in range(len(images)):
            filepath = os.path.join(self.images_folder, images[i])
            img = mpimg.imread(filepath)
            if (img.shape[0] != self.image_height) or (img.shape[1] != self.image_width):
                img = cv2.resize(img, (self.image_width, self.image_height))
            X[i,] = img
        return X

    def _get_Y(self, images):
        Y = np.empty((len(images), self.image_height, self.image_width, 4), dtype=np.int16)
        for i in range(len(images)):
            Y[i,] = get_image_masks(images[i], self.images_folder, self.segmentations_df, self.image_height,
                                    self.image_width, self.rle_height, self.rle_width)
        return Y