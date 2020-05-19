# ## Imports


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import Sequence
import cv2
from segmentation_models import Unet
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os
from sklearn.model_selection import train_test_split
import time


# > ## Load the data


labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
train_images_folder = '/home/matan/UnderstandingClouds/competition-data/images_augmented/'
#train_images_folder = '/kaggle/input/train-images-resized/train_images_320_480/'
train_images_files = [f for f in os.listdir(train_images_folder) if os.path.isfile(os.path.join(train_images_folder, f))]
train_df = pd.read_csv('/home/matan/UnderstandingClouds/competition-data/augmented_train_df.csv')
#train_df = pd.read_csv('/kaggle/input/train-resized/train_resized.csv')
test_images_folder = '/home/matan/UnderstandingClouds/competition-data/test_images/'
test_images_list = [file for file in os.listdir(test_images_folder) if os.path.isfile(os.path.join(test_images_folder, file))]
sample_submission_df = pd.read_csv('/home/matan/UnderstandingClouds/competition-data/sample_submission.csv')



filtered_train_images = list(filter(lambda x: '_' not in x, train_images_files))




#filtered_train_images = random.sample(filtered_train_images, 50)



# ## Functions for conversions between rle and mask represantations

def rle2mask(height, width, encoded):
    '''
    taken from https://github.com/pudae/kaggle-understanding-clouds
    '''
    if isinstance(encoded, float):
        img = np.zeros((height,width), dtype=np.uint8)
        return img

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


# In[9]:


def get_image_masks(filename, folder, segmentations_df, mask_height = 1400, mask_width = 2100, rle_height = 1400, rle_width = 2100):
    #segmentations are in run length format
    labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    filepath = os.path.join(folder, filename)
    image = mpimg.imread(filepath)
    masks = np.zeros((mask_height, mask_width, len(labels)))
    for i in range(4):
        segment = segmentations_df[segmentations_df.Image_Label == filename + '_' + labels[i]].EncodedPixels.values[0]
        mask = rle2mask(rle_height, rle_width, segment)
        if (mask.shape[0] != mask_height) or (mask.shape[1] != mask_width):
            mask = cv2.resize(mask, (mask_width, mask_height))
        masks[:, :, i] = mask
    return masks


# In[10]:


def mask_to_rle(img):
    '''
    taken from https://github.com/pudae/kaggle-understanding-clouds
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)    
        
def masks_to_rle(filename, masks):
    labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    seg_dict = {}
    for i in range(len(labels)):
        seg_dict[filename,'_',labels[i]] = mask_to_rle(masks[i])
    return pd.DataFrame(seg_dict.items(), columns = ['Image_Label', 'EncodedPixels'])


# ## Functions for plotting segmentations

# In[11]:


def get_segmentations(filename, df):
    return df[df.Image_Label.str.contains(filename)]



    

from keras.losses import  binary_crossentropy
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# In[16]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.dot(y_true_f, y_pred_f)
    return (2 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


# In[17]:


def dice_coef_tf(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# In[18]:


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


# ## Helper Functions

# In[19]:


def get_files_include_augmentations(file_list, augmentations, all_images):
    return list(filter(lambda x: (True in map(lambda y: y in x, augmentations) and x.split('_')[0] + '.jpg' in file_list) or x in file_list, all_images))


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
        curr_images = [self.images_list[i] for i in range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.images_list)))]
        X = self._get_X(curr_images)
        
        if not self.to_fit:
            return X
        
        Y = self._get_Y(curr_images)
        
        return X, Y
        
    def _get_X(self, images):
        X = np.empty((len(images), self.image_height, self.image_width, 3), dtype = np.int16)
        for i in range(len(images)):    
            filepath = os.path.join(self.images_folder, images[i])
            img = mpimg.imread(filepath)
            if (img.shape[0] != self.image_height) or (img.shape[1] != self.image_width):
                img = cv2.resize(img, (self.image_width, self.image_height))
            X[i,] = img
        return X
    
    def _get_Y(self, images):
        Y = np.empty((len(images), self.image_height, self.image_width, 4), dtype = np.int16)
        for i in range(len(images)):
            Y[i,] = get_image_masks(images[i], self.images_folder, self.segmentations_df, self.image_height, self.image_width, self.rle_height, self.rle_width)
        return Y    


# ## Plot metrics from a keras history object

# In[22]:


def plot_metrics(training_history):
    fig, a = plt.subplots(1,2)
    a[0].plot(training_history.history['loss'])
    a[0].plot(training_history.history['val_loss'])
    a[0].set_xlabel('epochs')
    a[0].set_ylabel('bce dice loss')

    a[1].plot(training_history.history['dice_coef_tf'])
    a[1].plot(training_history.history['val_dice_coef_tf'])
    a[1].set_xlabel('epochs')
    a[1].set_ylabel('dice coefficient')
    fig.tight_layout(pad = 3)
    fig.legend(['training', 'validation'], loc = 'upper center')


# ## Predict for given dataset with given model

# In[23]:


def predict_dataset(model, folder, file_list, resize = (320, 480), threshold = 0.5):
    datagen = DataGenerator(folder, file_list, None, 480, 320, 480, 320, labels, to_fit=False, batch_size=32)
    preds = np.empty((len(file_list), resize[0], resize[1], 4), dtype = np.int8)
    for i in range(datagen.__len__()):
        print(i)
        pred = model.predict(datagen.__getitem__(i))
        if resize != (320,480):
            pred_resized = [cv2.resize(pred[i,:,:,:], (resize[1],resize[0])) for i in range(pred.shape[0])]
            pred = np.asarray(pred_resized)
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        preds[i*32:(i+1)*32,:,:,:] = pred.astype(np.int8)
    
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
    
    masks_fish = preds[:,:,:,0]
    masks_flower = preds[:,:,:,1]
    masks_gravel = preds[:,:,:,2]
    masks_sugar = preds[:,:,:,3]
    
    all_masks = np.empty((len(file_list) * 4, resize[0], resize[1]), dtype = np.int8)
    all_masks[0::4] = masks_fish
    all_masks[1::4] = masks_flower
    all_masks[2::4] = masks_gravel
    all_masks[3::4] = masks_sugar

    rle_list = [mask_to_rle(all_masks[i]) for i in range(all_masks.shape[0])]
    res_df['EncodedPixels'] = rle_list
    return res_df, preds



def train(model_function,train_images, val_images, epochs, learning_rate = 0.001, augmentations = []):
    
    train_images = get_files_include_augmentations(train_images, augmentations, train_images_files)
    train_data_generator = DataGenerator(train_images_folder, train_images, train_df, 480, 320, 480, 320, labels, batch_size = 1)
    validation_data_generator = DataGenerator(train_images_folder, val_images, train_df, 480, 320, 480, 320, labels, batch_size = 1)
    # setup early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.01)
    
    #callbacks = [es]
    mc = ModelCheckpoint('best_model_unet_efficientnetb0' + str(int(time.time())) + '.h5', monitor='val_dice_coef_tf', mode='min', save_best_only=False)
    callbacks = [mc]
   # if model_checkpoint_callback is not None:
   #     callbacks.append(model_checkpoint_callback)
    unet_model = model_function()
    #unet_model.compile(optimizer=Adam(learning_rate), loss = 'binary_crossentropy', metrics = [dice_coef_tf])
    unet_model.compile(optimizer=Adam(learning_rate), loss = bce_dice_loss, metrics = [dice_coef_tf])
    #unet_model.compile(optimizer='adam', loss = bce_dice_loss, metrics = [dice_coef_tf])

    unet_model.summary()
    history = unet_model.fit_generator(train_data_generator, validation_data = validation_data_generator, epochs = epochs, callbacks = callbacks)
    return unet_model, history
    


# >  ### Train on Unet with efficientnetb0 backbone and different augmentations ###

# In[29]:


def unet_efficientnet():
    model = Unet('efficientnetb0', input_shape=(320, 480, 3), encoder_weights='imagenet', classes = 4, encoder_freeze = False)
    return model


# In[30]:


train_images, val_images = train_test_split(filtered_train_images, train_size = 0.80)


model, metrics_history = train(unet_efficientnet,  train_images, val_images, epochs = 3, learning_rate = 0.0001, augmentations = ['vertical', 'horizontal', 'rotated_45'])
plot_metrics(metrics_history)


#test_df, test_masks = predict_dataset(model, test_images_folder, test_images_list[:200])


# In[ ]:


#import random
#images_to_plot = random.sample(test_images_list[:200], 10)
#for image in images_to_plot:
#    plot_image_with_rle(image, test_images_folder, test_df)




#sub_df, history = predict_dataset(model, test_images_folder, test_images_list, resize = (350, 525), threshold=0.5)



#sub_df = sub_df[['Image_Label', 'EncodedPixels']]

#sub_df.set_index('Image_Label', inplace = True)

#sub_df.to_csv('submission.csv')
