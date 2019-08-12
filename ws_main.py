import model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from visual_callbacks import AccLossPlotter
from tensorflow.python.client import device_lib

import numpy as np
import datetime
import json


BATCH_SIZE = 512
NUM_EPOCHS = 100

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
    
    sn.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])
    print(sn.summary)

    #training
    training_dir = '/kw_resources/food/dataset/training_data/'
    #training_dir = "../training_data/"
    validation_dir = '/kw_resources/food/dataset/testing_data/'
    #validation_dir = "../testing_data/"
    num_training = 166580  #use find . -type f | wc -l for each directory
    num_validation = 60990
    num_epochs = NUM_EPOCHS

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

    sn.fit_generator(
        train_data,
        steps_per_epoch=(num_training // BATCH_SIZE),
        epochs = num_epochs,
        validation_data=validation_data_generator,
        validation_steps=(num_validation // BATCH_SIZE))

    history = sn
    with open('food/results/{}.json'.format(datetime.datetime.now().strftime('%m-%d-%X')), 'w') as f:
        json.dump(history.history, f)

   # sn.save_weights('/kw_resources/food/results/weights.h5')

if __name__ == '__main__':
    main()
    input('Press ENTER to exit..')
