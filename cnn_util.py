#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:42:19 2018

@author: guillaume
"""

import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


def print_file_layers(filename):
    """
    show layers stored in a h5 file
    
    filename: h5 file to open
    """
    #Print layers in the file

    with h5py.File(filename) as f:
        #print(list(f))
        for i in list(f):
            print(i)

# Compile and train the CNN
def train_classifier(classifier, new_epochs, batch_size=32, verbose=1,
                     use_checkpoints=False, initial_epoch=0,
                     train_size=None, test_size=None):
    classifier.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    
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
                                                batch_size=batch_size,
                                                class_mode='binary')
    

    # Add checkpoints to save weights in case the test set acc improved
    if use_checkpoints:    
        filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
    else:
        callbacks_list = None
    # Fit the model
    #model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
    
    #TODO check the train and test folder to provide default values if needed
    if train_size is None:
        train_size = 8000
    if test_size is None:
        test_size = 2000
        
    history = classifier.fit_generator(training_set,
                         steps_per_epoch=train_size,
                         epochs=initial_epoch + new_epochs,
                         validation_data=test_set,
                         validation_steps=test_size,
                         initial_epoch=initial_epoch,
                         verbose=verbose,
                         callbacks=callbacks_list)
    classifier.save_weights("classifier2_1.h5")
    return history

if __name__ == "__main__":
    filename = "classifier_final.h5"
    print_file_layers(filename)
