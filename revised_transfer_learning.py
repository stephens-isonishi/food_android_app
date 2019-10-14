import os, sys
#import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import argparse
import numpy as np
import datetime
import platform
import glob


# from IPython.display import display
# from PIL import Image
from keras import backend as K
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from PIL import Image

#WIDTH = 299, HEIGHT = 299
NUM_EPOCHS = 1
BATCH_SIZE = 32
NUM_CLASSES = 451
TRAINING_DIR = '/kw_resources/food/dataset/training_data/'
TESTING_DIR = '/kw_resources/food/dataset/testing_data/'
FILEPATH = '/kw_resources/food/transfer_learning_training/'
# FILEPATH = '../history_training/'
# DIR1 = '../training_data/'
# DIR2 = '../testing_data/'

#found using: find DIR_NAME -type f | wc -l       --from stack overflow
TRAIN_SIZE = 166580
TEST_SIZE = 60990

def find_most_recent_model(directory):
	if not os.listdir(directory):
		return '', 0
	list_of_files = glob.glob(directory + '*') #* means all 
	latest_file = max(list_of_files, key=os.path.getctime) #get most recent file
	count = len([1 for x in list(os.scandir(directory)) if x.is_file()])
	return latest_file, count

def clean_directory(directory):
	source = directory
	for files in os.listdir(source):
		file_path = os.path.join(source, files)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e)



def last_layer_insertion(base_model, num_classes):
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(4096)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(.5)(x)
	predictions = Dense(num_classes, activation='softmax')(x)
	model = Model(input=base_model.input, output=predictions)
	return model

def main(args):
    print("started program...")
    print(platform.python_version())

    num_epochs = int(args.nb_epoch)
    batch = BATCH_SIZE
    num_classes = NUM_CLASSES

    # training_dir = TRAINING_DIR
    # testing_dir = TESTING_DIR
    training_dir = TRAINING_DIR
    testing_dir = TESTING_DIR
    num_training = TRAIN_SIZE
    num_testing = TEST_SIZE

    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.1,
        rescale=1./255)

    test_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.1,
        rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(299,299),
        batch_size=batch,
        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        testing_dir,
        target_size=(299,299),
        batch_size=batch,
        class_mode='categorical')



    base_model = InceptionV3(
    	weights='imagenet',
    	include_top=False, 
    	input_tensor=Input(shape=(299,299,3)))

    model = last_layer_insertion(base_model, num_classes)

    for layer in model.layers[:172]:
    	layer.trainable = False
    for layer in model.layers[172:]:
    	layer.trainable = False



    num_files = 0

    if args.new_training:
    	print('deleting previous training history...')
    	clean_directory(FILEPATH)
    saved_model, current_epoch_num = find_most_recent_model(FILEPATH)

    print("current epoch: {}".format(current_epoch_num))

    if saved_model.endswith('.hdf5') and current_epoch_num != 0:
    	model = load_model(saved_model)


    #try adam too...
    model.compile(
    	optimizer=SGD(lr=0.0001, momentum=0.9), 
    	loss='categorical_crossentropy',
    	metrics=['accuracy'])
    print("compiled successfully...")

    if current_epoch_num > num_epochs:
    	print('already trained for {} epochs'.format(current_epoch_num))
    	exit()
    else:
    	num_epochs_togo = num_epochs - current_epoch_num
    	print('trained for {} epochs so far, {} more epochs to go...'.format(current_epoch_num, num_epochs_togo))


    filepath = FILEPATH + "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    
    checkpoint = ModelCheckpoint(
    	filepath,
    	monitor='val_acc',
    	verbose=1,
    	save_best_only=False,
    	save_weights_only=False,
    	mode='max')

    csv_logger = CSVLogger(FILEPATH + 'epochs_training.log')

    model.fit_generator(
    	train_generator,
    	validation_data=test_generator,
    	steps_per_epoch=(num_training // batch),
    	epochs=num_epochs,
    	validation_steps=(num_testing // batch),
    	callbacks=[csv_logger, checkpoint],
    	verbose=1)

    print("done")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--nb_epoch", "-e", default=NUM_EPOCHS)
    args.add_argument('--new_training', '-n', action='store_true')
    args = args.parse_args()
    main(args)