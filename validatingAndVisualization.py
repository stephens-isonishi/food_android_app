'''
validation and visualization of model results so it wouldn't be a "black box"
'''


import os, sys
#import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime
import platform
import glob
import shutil
import tensorflow as tf

# from IPython.display import display
# from PIL import Image
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


def predict_crop(img, img_num, show_images=True, debugging=True):
	

def main():
    model = load_model(WEIGHT_FILE)


if __name__ == '__main__':
	main()