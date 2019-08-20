import model
from keras.optimizers import SGD
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
import re


FILEPATH = '/kw_resources/food/model_weights/'
BATCH_SIZE = 512
NUM_EPOCHS = 1

def find_most_recent_model():
    recent = 0
    ans = ""
    for file in os.listdir(FILEPATH):
        temp = int(re.search(r'\d+', file).group(0))
        if recent < temp:
            recent = temp
            ans = file
    return ans, recent

def main():
    np.random.seed(45)
    nb_class = 451 #found by doing: echo */ | wc
    width, height = 224, 224

    sn = model.SqueezeNet(nb_classes=nb_class, inputs=(3, height, width))

    local_devices = device_lib.list_local_devices()
    num_gpus = len([dev.name for dev in local_devices if dev.device_type == 'GPU'])
    print(num_gpus)
    if(num_gpus >= 2):
        sn = multi_gpu_model(sn, num_gpus)
    print('build model')

    #obviously mess around with the hyperparameters
    #sgd = SGD(lr = .001, decay=.0002, momentum=.9, nesterov=True)
    current_epoch_num = 0
    saved_model, current_epoch_num = find_most_recent_model()
    if saved_model.is_file():
        sn.load(saved_model) ## or is it just load? according keras callbacks

    ######question is, does load_model(saved_model) save the number of epochs?
    ###does it matter? we just look at number, add everything?

    sn.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])
    print(sn.summary)

    #training
    training_dir = '/kw_resources/food/dataset/training_data/'
    #training_dir = "../training_data/"
    validation_dir = '/kw_resources/food/dataset/testing_data/'
    #validation_dir = "../testing_data/"
    num_training = 166580  #use find . -type f | wc -l for each directory
    num_validation = 60990
    num_epochs = 1
    if current_epoch_num > NUM_EPOCHS:
        print("already trained for {} epochs".format(current_epoch_num))
        exit()
    else:
        num_epochs = NUM_EPOCHS - current_epoch_num
        print("number of epochs to train for: {}".format(num_epochs))


    #generation
    training_generator_parameters = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
    testing_generator_parameters = ImageDataGenerator(rescale=1./255)
    train_data = training_generator_parameters.flow_from_directory(
        training_dir,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    validation_data_generator = testing_generator_parameters.flow_from_directory(
        validation_dir,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
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

    sn.fit_generator(
        train_data,
        steps_per_epoch=(num_training // BATCH_SIZE),
        epochs = num_epochs,
        validation_data=validation_data_generator,
        validation_steps=(num_validation // BATCH_SIZE),
        callbacks=callbacks_list,
        verbose=1)

    history = sn
    with open('/kw_resources/food/results/e:{}_b:{}_{}'.format(num_epochs, batch_size, datetime.datetime.now().strftime('%m-%d-%X')), 'wb') as f:
        pickle.dump(history.history, f)

   # sn.save_weights('/kw_resources/food/results/weights.h5')

if __name__ == '__main__':
    main()
    input('Press ENTER to exit..')
