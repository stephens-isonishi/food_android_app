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
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from PIL import Image

MODEL_FILE = '../history_training/7/weights-36-0.86.hdf5'
BATCH = 32

#testing generator used for training. 
def testingGenerator(testing_dir, batch):
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
        testing_dir,
        target_size=(299,299),
        batch_size=batch,
        shuffle=False,
        class_mode='categorical')
	return test_generator


def main():
	batch = BATCH
	model = load_model(MODEL_FILE)
	test_generator = testingGenerator('../testing_data/', batch)
	labels_file = 'labels.txt'
    


if __name__ == '__main__':
	main()