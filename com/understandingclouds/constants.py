import argparse
import sys

test_locally = False

if test_locally:
    DATA_DIR = '/home/matan/UnderstandingClouds/competition-data/'
else:
    DATA_DIR = ''
LABELS = ['Fish', 'Flower', 'Gravel', 'Sugar']
TRAIN_IMAGES_FOLDER = DATA_DIR + 'train_images/'
TEST_IMAGES_FOLDER = DATA_DIR + 'test_images/'
AUGMENTED_IMAGES_FOLDER = DATA_DIR + 'images_augmented/'
TRAIN_DF_FILE = DATA_DIR + 'augmented_train_df.csv'
