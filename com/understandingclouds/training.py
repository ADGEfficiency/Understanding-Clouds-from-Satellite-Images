from utils import get_files_include_augmentations
from data_generator import DataGenerator
from constants import TRAIN_DF_FILE, LABELS, AUGMENTED_IMAGES_FOLDER
from metrics import dice_coef_tf, bce_dice_loss
import time
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from segmentation_models import Unet
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


def train(model_function, train_images, val_images, epochs, learning_rate=0.001, augmentations=[]):
    train_df = pd.read_csv(TRAIN_DF_FILE)
    train_images = get_files_include_augmentations(train_images, augmentations, train_images_files)
    train_data_generator = DataGenerator(AUGMENTED_IMAGES_FOLDER, train_images, train_df, 480, 320, 480, 320, LABELS,
                                         batch_size=16)
    validation_data_generator = DataGenerator(AUGMENTED_IMAGES_FOLDER, val_images, train_df, 480, 320, 480, 320, LABELS,
                                              batch_size=16)
    # setup early stopping
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.01)

    # callbacks = [es]
    mc = ModelCheckpoint('best_model_unet_efficientnetb0' + str(int(time.time())) + '.h5', monitor='val_dice_coef_tf',
                         mode='min', save_best_only=False)
    callbacks = [mc]
    # if model_checkpoint_callback is not None:
    #     callbacks.append(model_checkpoint_callback)
    unet_model = model_function()
    # unet_model.compile(optimizer=Adam(learning_rate), loss = 'binary_crossentropy', metrics = [dice_coef_tf])
    unet_model.compile(optimizer=Adam(learning_rate), loss=bce_dice_loss, metrics=[dice_coef_tf])
    # unet_model.compile(optimizer='adam', loss = bce_dice_loss, metrics = [dice_coef_tf])

    unet_model.summary()
    history = unet_model.fit_generator(train_data_generator, validation_data=validation_data_generator, epochs=epochs,
                                       callbacks=callbacks)
    return unet_model, history


def unet_efficientnet():
    model = Unet('efficientnetb0', input_shape=(320, 480, 3), encoder_weights='imagenet', classes=4,
                 encoder_freeze=False)
    return model


def plot_metrics(training_history):
    fig, a = plt.subplots(1, 2)
    a[0].plot(training_history.history['loss'])
    a[0].plot(training_history.history['val_loss'])
    a[0].set_xlabel('epochs')
    a[0].set_ylabel('bce dice loss')

    a[1].plot(training_history.history['dice_coef_tf'])
    a[1].plot(training_history.history['val_dice_coef_tf'])
    a[1].set_xlabel('epochs')
    a[1].set_ylabel('dice coefficient')
    fig.tight_layout(pad=3)
    fig.legend(['training', 'validation'], loc='upper center')
    plt.show()


train_images_files = [f for f in os.listdir(AUGMENTED_IMAGES_FOLDER) if
                      os.path.isfile(os.path.join(AUGMENTED_IMAGES_FOLDER, f))]
filtered_train_images = list(filter(lambda x: '_' not in x, train_images_files))
#filtered_train_images = filtered_train_images[0:10]
train_images, val_images = train_test_split(filtered_train_images, train_size=0.80)
print('num of training samples is {}'.format(len(train_images)))
print('num of val samples is {}'.format(len(val_images)))
model, metrics_history = train(unet_efficientnet, train_images, val_images, epochs=3, learning_rate=0.0001,
                               augmentations=['vertical', 'horizontal', 'rotated_45'])
plot_metrics(metrics_history)
