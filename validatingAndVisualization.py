'''
validation and visualization of model results so it wouldn't be a "black box"
'''


import os, sys
#import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime
import platform
import cv2
# import tensorflow as tf

# from IPython.display import display
from PIL import Image
from keras import backend as K
from keras import regularizers
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model


WEIGHT_FILE = 'weight_files/weights-50-0.87.hdf5'

def center_crop(image, center_crop_size):
    centerw, centerh = image.shape[0] // 2, image.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return image[centerw - halfw: centerw + halfw + 1, centerh - halfh: centerh + halfh + 1, :]


def predict_crop(model, img, actual_label, show_images=True, debugging=True):

	res = cv2.resize(img, dsize=(299,299), interpolation=cv2.INTER_CUBIC)

	crops = [
	    img[:299,:299, :], # Upper Left
        img[:299, img.shape[1]-299:, :], # Upper Right
        img[img.shape[0]-299:, :299, :], # Lower Left
        img[img.shape[0]-299:, img.shape[1]-299:, :], # Lower Right
        center_crop(img, (299, 299)), #center region
        res #downscaled image to 299 x 299 size
	]
	if show_images:
		figure, axes = plt.subplots(2,3, figsize=(8,4))
		#figure.delaxes(axes[1,2])
		axes[0][0].imshow(crops[0])
		axes[0][1].imshow(crops[1])
		axes[0][2].imshow(crops[2])
		axes[1][0].imshow(crops[3])
		axes[1][1].imshow(crops[4])
		axes[1][2].imshow(crops[5])


	y_preds = model.predict(np.array(crops)) 
	top_pred = np.argmax(y_preds, axis=1)
	top_5_preds = np.argpartition(y_preds, -5)[:,-5:]

	if debugging:
		print('top 1 predictions:', top_pred)
		print('top 5 predictions:', top_5_preds)
		print('true label:', actual_label)



def main():
    model = load_model(WEIGHT_FILE)
    image_name = '../testing_data/pizza/2003290.jpg'
    image = np.array(Image.open(image_name))

    ### actual label is not available currently, due to shuffling data with ImageDataGenerator in revised_transfer_learning.py
    ### may need to retrain to get this data and generate the confusion matrix and whatnot
    
    predict_crop(model,image, 'xyz')

if __name__ == '__main__':
	main()
