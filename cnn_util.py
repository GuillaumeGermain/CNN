#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:42:19 2018

@author: guillaume
"""

import h5py
from keras.preprocessing.image import ImageDataGenerator


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
def train_classifier(classifier, epochs, batch_size=32, verbose=1):
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
    
    history = classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=epochs,
                         validation_data=test_set,
                         validation_steps=2000,
                         verbose=verbose)
    classifier.save_weights("classifier2_1.h5")
    return history

if __name__ == "__main__":
    filename = "classifier_final.h5"
    print_file_layers(filename)
