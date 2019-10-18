'''
This code was unusable on the lab computer because it loads all images into a giant numpy array and there is not enough memory for it. One of the benefits of using a numpy array is the ability to employ sklearn train-test-split at ratios we like. However, the creators of the food dataset (Food-475) already split the data into training and testing so the other method of flow_from_directory is better in this case. See revised_transfer_learning.py
'''



import os, sys
#import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import argparse
import numpy as np

# from IPython.display import display
# from PIL import Image
from keras import backend as K
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from PIL import Image
from sklearn.model_selection import train_test_split

#WIDTH = 299, HEIGHT = 299
NUM_EPOCHS = 1
BATCH_SIZE = 32
NUM_CLASSES = 471
TRAINING_DIR = '/kw_resources/food/dataset/training_data/'
TESTING_DIR = '/kw_resources/food/dataset/testing_data/'
FILEPATH = '/kw_resources/food/transfer_learning_training/'
DIR1 = '../training_data/'
DIR2 = '../testing_data/'

def load_images_into_np(dir1, dir2):
	#X = np.array([,,,])
	X = list()
	y = list()
	directories = [dir1, dir2]
	count = 0
	for i in directories:
		for root, dirs, files in os.walk(i):
			for name in files:
				image_path = os.path.join(root, name)
				temp = root
				temp = temp.split('/')
				label = temp[len(temp)-1]
				image_pixels = list(Image.open(image_path).getdata())
				#X = np.vstack((X, image_pixels))
				X.append(image_pixels)
				y.append(label)
				count = count + 1
				if count%1000 == 0:
					print(count)

			else:
				continue
			break

	X = np.array(X)
	print(X.shape)
	return X, y

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
    num_epochs = int(args.nb_epoch)
    batch = BATCH_SIZE
    num_classes = NUM_CLASSES
    X, y = load_images_into_np(TRAINING_DIR, TESTING_DIR)


    print("loaded into giant numpy array")

    #default split is test size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("split training,testing")
    num_training = X_train[0]
    num_testing = X_test[0]

    print("number of training samples: {}".format(num_training))
    print("number of testing samples: {}".format(num_testing))

    y_train_cat = np_utils.to_categorical(y_train, num_classes)
    y_test_cat = np_utils.to_categorical(y_test, num_classes)


    datagen = ImageDataGenerator(
    	featurewise_center=False,
    	samplewise_center=False,
    	featurewise_std_normalization=False,
    	zca_whitening=False,
    	rotation_range=45,
    	width_shift_range=0.125,
    	height_shift_range=0.125,
    	horizontal_flip=True,
    	vertical_flip=False,
    	rescale=1./255)

    datagen.fit(X_train)
    generator = datagen.flow(X_train, y_train_cat, batch_size=batch)
    val_generator = datagen.flow(X_test, y_test_cat, batch_size=batch)
    print("set up image data generator")

    base_model = InceptionV3(
    	weights='imagenet',
    	include_top=False, 
    	input_tensor=Input(shape=(299,299,3)))
    print("loaded inceptionv3...")
    model = last_layer_insertion(base_model, num_classes)
    print("inserted last layer...")
    for layer in model.layers[:172]:
    	layer.trainable = False
    for layer in model.layers[172:]:
    	layer.trainable = False

    model.compile(
    	optimizer=SGD(lr=0.0001, momentum=0.9), 
    	loss='categorical_crossentropy',
    	metrics=['accuracy'])
    print("compiled successfully...")
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
    	generator,
    	validation_data=val_generator,
    	steps_per_epoch=(num_training // batch),
    	epochs=num_epochs,
    	validation_steps=(num_testing // batch),
    	callbacks=[csv_logger, checkpoint],
    	verbose=1)

    print("done")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--nb_epoch", "-e", default=NUM_EPOCHS)
    args = args.parse_args()
    main(args)
