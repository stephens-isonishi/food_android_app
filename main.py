import model
from keras.optimizers import SGD
#from keras.utils import multi_gpu_model 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from visual_callbacks import AccLossPlotter
from tensorflow.python.client import device_lib

import numpy as np
import os
import pickle
import datetime
import json
import argparse
import tensorflow as tf


BATCH_SIZE = 64
NUM_EPOCHS = 1

import warnings
warnings.filterwarnings("ignore")

def main(args):


    nb_class = 451 #found by doing: echo */ | wc
    width, height = 224, 224

    sn = model.SqueezeNet(nb_classes=nb_class, inputs=(3, height, width))
    # local_devices = device_lib.list_local_devices()
    # num_gpus = len([dev.name for dev in local_devices if dev.device_type == 'GPU'])
    # print(num_gpus)
    # if(num_gpus >= 2):
    #     sn = multi_gpu_model(sn, num_gpus)
    print('build model')

    #obviously mess around with the hyperparameters
    sgd = SGD(lr = .001, decay=.0002, momentum=.9, nesterov=True)
    
    #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    #runmeta = tf.RunMetadata()
    #add run_metadata=runmeta inside compile....    
    #sn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics =['accuracy'], options = run_opts)
    sn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics =['accuracy'])


    print(sn.summary)

    #training
    #training_dir = '/kw_resources/food/dataset/training_data/'
    training_dir = "../training_data/"
    #validation_dir = '/kw_resources/food/dataset/testing_data/'
    validation_dir = "../testing_data/"
    num_training = 166580  #use find . -type f | wc -l for each directory
    num_validation = 60990
    num_epochs = int(args.nb_epochs)
    bat_size = int(args.batch_size)

    #generation
    training_generator_parameters = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
    testing_generator_parameters = ImageDataGenerator(rescale=1./255)
    train_data = training_generator_parameters.flow_from_directory(
        training_dir,
        target_size=(width, height),
        batch_size=bat_size,
        class_mode='categorical')

    validation_data_generator = testing_generator_parameters.flow_from_directory(
        validation_dir,
        target_size=(width, height),
        batch_size=bat_size,
        class_mode='categorical')
    print("before fit gen")
    sn.fit_generator(
        train_data,
        steps_per_epoch=(num_training // bat_size),
        epochs=num_epochs,
        workers=0,
        use_multiprocessing=False,
        validation_data=validation_data_generator,
        validation_steps=(num_validation // bat_size))
    print("fit_gen was not issue")
    history = sn
    with open('../training_hist/e:{}_b:{}_{}'.format(num_epochs, bat_size,datetime.datetime.now().strftime('%m-%d-%X')), 'wb') as f:
        pickle.dump(history.history, f)

   # sn.save_weights('/kw_resources/food/results/weights.h5')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--nb_epochs", default=NUM_EPOCHS)
    args.add_argument("--batch_size", default=BATCH_SIZE)
    args = args.parse_args()
    main(args)
    input('Press ENTER to exit..')
