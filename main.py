import model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from visual_callbacks import AccLossPlotter
import numpy as np

def main():
    np.random.seed(45)
    nb_class = 451 #found by doing: echo */ | wc
    width, height = 224, 224

    sn = model.SqueezeNet(nb_classes=nb_class, inputs=(3, height, width))
    print('build model')

    #obviously mess around with the hyperparameters
    sgd = SGD(lr = .001, decay=.0002, momentum=.9, nesterov=True)
    
    sn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics =['accuracy'])
    print(sn.summary)

    #training
    training_dir = '../training_data/'
    validation_dir = '../testing_data/'
    num_training = 166580  #use find . -type f | wc -l for each directory
    num_validation = 60990
    num_epochs = 10

    #generation
    training_generator_parameters = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
    testing_generator_parameters = ImageDataGenerator(rescale=1./255)
    train_data = training_generator_parameters.flow_from_directory(
        training_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')

    validation_data_generator = testing_generator_parameters.flow_from_directory(
        validation_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')

    sn.fit_generator(
        train_data,
        samples_per_epoch=num_training,
        validation_data=validation_data_generator,
        nb_val_samples=num_validation)


if __name__ == '__main__':
    main()
    input('Press ENTER to exit..')
