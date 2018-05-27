#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:42:19 2018

@author: guillaume
"""

import h5py

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback#, ReduceLROnPlateau
from keras.optimizers import Adam

import keras.backend as K
from keras.regularizers import l2


# Are GPUZ available?
from tensorflow.python.client import device_lib
def get_available_gpus():
    """
        Usage: gpus = get_available_gpus()
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



def print_file_layers(filename):
    """
    show layers stored in a h5 file
    filename: h5 file to open
    """
    print(filename)
    with h5py.File(filename) as f:
        #print(list(f))
        for i in list(f):
            print(i)

def print_model_layers(model):
    # classifier 1 layers

    print(model.model.name + " layers")
    for layer in model.layers:
        print(layer.name.ljust(20), type(layer))



def build_model(num_conv_layers=1, name=None):
    model = Sequential()
    if name is not None:
        model.name = name
    else:
        model.name = "sequential_" + str(num_conv_layers)
    
    for i in range(num_conv_layers):
        model.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
    return model


def build_model_2(num_conv_layers=1, name=None):
    model = Sequential()
    if name is not None:
        model.name = name
    else:
        model.name = "sequential_" + str(num_conv_layers)
    
    for i in range(num_conv_layers):
        model.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
    return model



### Efficient model to try
def build_model_eff():
    model = Sequential()
    model.name = "sequential_eff"
    
    model.add(Convolution2D(32, kernel_size=(5,5), padding='valid', input_shape=(64,64,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    
    model.add(Convolution2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='relu', kernel_regularizer=l2(0.01)))
    
    return model





# New Callback descendant to print the learning rate after each epoch
class Callback_show_learn_param(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        
        # Beta values
        beta_1=self.model.optimizer.beta_1
        beta_2=self.model.optimizer.beta_2
        print("lr", K.eval(lr), "decay", K.eval(decay), "lr_with_decay", K.eval(lr_with_decay),
              "beta_1", K.eval(beta_1), "beta_2", K.eval(beta_2))
        
#        # other parameters
#        clipnorm = self.model.optimizer.clipnorm
#        clipvalue = self.model.optimizer.clipvalue
#        print("clipnorm", K.eval(clipnorm), "clipvalue", K.eval(clipvalue))
        
def show_optimizer_settings(model):
    lr = model.optimizer.lr
    decay = model.optimizer.decay
    print("lr", K.eval(lr), "decay", K.eval(decay))
    
    # Beta values
    beta_1=model.optimizer.beta_1
    beta_2=model.optimizer.beta_2
    print("beta_1", K.eval(beta_1), "beta_2", K.eval(beta_2))
#    # other parameters
#    clipnorm = self.model.optimizer.clipnorm
#    clipvalue = self.model.optimizer.clipvalue
#    print("clipnorm", K.eval(clipnorm), "clipvalue", K.eval(clipvalue))
    


# Compile and train the CNN
def train_model(model, new_epochs, initial_epoch=0,
                train_size=None, test_size=None,
                batch_size=16, verbose=1
                ):

    # Set Data Generators for training and test sets    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=batch_size,
                                                     class_mode='binary')
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=1, #max(batch_size // 4, 1),
                                                class_mode='binary')
    
    #TODO check the train and test folder to provide default values if needed
    if train_size is None:
        train_size = 8000
    if test_size is None:
        test_size = 2000
    
    # Use existing optimizer if already defined else use Adam
    try:
        optimizer = model.optimizer
        print("Optimizer: found in model")
    except:
        optimizer = Adam(lr=0.001)
        print("Optimizer not found in model: Using Adam(lr=0.001)")

        
    # Manage callbacks for debugging
    callback_list = []
    metric_list = ['accuracy']
    
    # Add checkpoints to save weights in case the test set acc improved
    filepath = "models/" + model.name + "-weights-improvement-{epoch:02d}-{val_acc:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max', verbose=0)
    callback_list.append(checkpoint)

#    # Reduce LR on plateau by default
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                          patience=5, min_lr=1.e-5)
#    callback_list.append(reduce_lr)
        
    model.compile('adam', loss='binary_crossentropy', metrics=metric_list)

    history = model.fit_generator(training_set,
                         steps_per_epoch=train_size,
                         epochs=initial_epoch + new_epochs,
                         validation_data=test_set,
                         validation_steps=test_size,
                         initial_epoch=initial_epoch,
                         verbose=verbose,
                         callbacks=callback_list)
    model.save_weights("classifier2_1.h5")
    return history

if __name__ == "__main__":
    filename = "classifier_final.h5"
    print_file_layers(filename)
