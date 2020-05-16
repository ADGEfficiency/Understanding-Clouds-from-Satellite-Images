import random
from com.understandingclouds.plotting import plot_image_with_rle, get_segmentations
from com.understandingclouds.constants import TRAIN_IMAGES_FOLDER, TRAIN_DF_FILE
from com.understandingclouds import constants
import os
import pandas as pd
import argparse
import sys

def plot_random_train_segmentations():
    print('plotting train images...')
    train_images_files = [f for f in os.listdir(TRAIN_IMAGES_FOLDER) if
                          os.path.isfile(os.path.join(TRAIN_IMAGES_FOLDER, f))]
    train_df = pd.read_csv(TRAIN_DF_FILE)
    # plot segmentations of 10 random images
    images_to_plot = random.sample(train_images_files, 10)
    for image in images_to_plot:
        plot_image_with_rle(image, TRAIN_IMAGES_FOLDER, get_segmentations(image, train_df))


parser = argparse.ArgumentParser()
mutex_group = parser.add_mutually_exclusive_group()
mutex_group.add_argument('--plot-train-segmentations', action='store_true')
options = vars(parser.parse_args(sys.argv[1:]))
if options['plot_train_segmentations']:
    plot_random_train_segmentations()
