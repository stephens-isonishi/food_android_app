import model
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import multi_gpu_model 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from visual_callbacks import AccLossPlotter
from tensorflow.python.client import device_lib
from pathlib import Path

import numpy as np
import pickle
import datetime
import json
import os
import shutil
import glob
import argparse



FILEPATH = '/kw_resources/food/model_weights/'
BATCH_SIZE = 256
NUM_EPOCHS = 200

def empty_folder():
    source = FILEPATH
    num_epochs = len([1 for x in list(os.scandir(source)) if x.is_file])
    dest1 = '/kw_resources/food/weight_history/epochs{}'.format(num_epochs)

    files = os.listdir(source)
    for f in files:
        shutil.move(source+f, dest1)



def find_most_recent_model():
    if not os.listdir(FILEPATH):
        return "", 0
    list_of_files = glob.glob(FILEPATH + '*') # * means all 
    latest_file = max(list_of_files, key=os.path.getctime) #could be min depending on os

    count = len([1 for x in list(os.scandir(FILEPATH)) if x.is_file()])
    return latest_file, count

def main(args):
    reset_training = args.reset_training
    if reset_training:
        "resetting training!"
        empty_folder()

    np.random.seed(45)
    nb_class = 451 #found by doing: echo */ | wc
    width, height = 224, 224
    bat_size = BATCH_SIZE

    #structure
    sn = model.SqueezeNet(nb_classes=nb_class, inputs=(3, height, width))

    #multi-gpu
    local_devices = device_lib.list_local_devices()
    num_gpus = len([dev.name for dev in local_devices if dev.device_type == 'GPU'])
    print(num_gpus)
    if(num_gpus >= 2):
        sn = multi_gpu_model(sn, num_gpus)
    print('build model')



    #obviously mess around with the hyperparameters
    #sgd = SGD(lr = .001, decay=.0002, momentum=.9, nesterov=True)

    num_files = 0
    saved_model, current_epoch_num = find_most_recent_model()
    print("saved model: " + saved_model + "current epoch: {}".format(current_epoch_num))
    if len(saved_model) > 0 and current_epoch_num != 0:
        sn = load_model(saved_model)


    ##potential issue: adam may reset previous history, may need to use a different optimizer
    sn.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])
    print(sn.summary)


    training_dir = '/kw_resources/food/dataset/training_data/'
    validation_dir = '/kw_resources/food/dataset/testing_data/'
    
    num_training = 166580  #used find . -type f | wc -l for each directory
    num_validation = 60990
    
    num_epochs = int(args.nb_epochs)
    bat_size = int(args.batch_size)
    
    if current_epoch_num > num_epochs:
        print("already trained for {} epochs".format(current_epoch_num))
        exit()
    else:
        num_epochs = num_epochs - current_epoch_num
        print("number of epochs to train for: {}".format(num_epochs))



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


    #checkpoint
    filepath = FILEPATH + "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=False, 
        save_weights_only=False, 
        mode='max')
    callbacks_list = [checkpoint]

    #fitting
    sn.fit_generator(
        train_data,
        steps_per_epoch=(num_training // bat_size),
        epochs = num_epochs,
        validation_data=validation_data_generator,
        validation_steps=(num_validation // bat_size),
        callbacks=callbacks_list,
        verbose=1)

    #save history to use at a later time
    history = sn
    with open('/kw_resources/food/results/e:{}_b:{}_{}'.format(num_epochs, bat_size, datetime.datetime.now().strftime('%m-%d-%X')), 'wb') as f:
        pickle.dump(history.history, f)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--nb_epochs", default=NUM_EPOCHS)
    args.add_argument("--batch_size", default=BATCH_SIZE)
    args.add_argument("--reset_training",'-r', action='store_true')

    args = args.parse_args()
    main(args)
