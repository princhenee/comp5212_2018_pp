import os
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, ELU, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K
from config import *
from load_data import generate_data_batch, split_train_val

def mkdir_p(folder_name):
    ''' mkdir -p folder_name
    '''
    if os.path.isdir(folder_name) == False:
        os.mkdir(folder_name)

def get_nvidia_model(summary=True):
    """ Get the keras Model corresponding to the NVIDIA architecture described in:
    Bojarski, Mariusz, et al. "End to end learning for self-driving cars."

    :param summary: show model summary
    :return: keras Model of NVIDIA architecture
    """
    init = 'glorot_uniform'

    # input_frame = Input(shape=(NVIDIA_H, NVIDIA_W))

    # standardize input

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(1, NVIDIA_H, NVIDIA_W))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2), kernel_initializer="glorot_uniform"))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2), kernel_initializer="glorot_uniform"))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2), kernel_initializer="glorot_uniform"))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform"))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    if summary:
        model.summary()

    return model


if __name__ == '__main__':

    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='training_data/driving_log.csv')

    # get network model and compile it (default Adam opt)
    nvidia_net = get_nvidia_model(summary=True)
    nvidia_net.compile(optimizer='adam', loss='mse')

    # json dump of model architecture
    mkdir_p("logs")
    with open('logs/model.json', 'w') as f:
        f.write(nvidia_net.to_json())

    # define callbacks to save history and weights
    mkdir_p("checkpoints")
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/history.csv')

    # start the training
    nvidia_net.fit_generator(generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']),
                         steps_per_epoch=300,
                         epochs=50,
                         validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0),
                         validation_steps=100,
                         callbacks=[checkpointer, logger])
