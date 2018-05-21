#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:55:27 2018

@author: guillaume

New script for Sasha

"""


# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Convolution2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense

from cnn_util import print_file_layers, train_classifier


filename = "classifier_1_epoch_45.h5"
#print_file_layers(filename)


# Load the model from the dead
from keras.models import load_model
model = load_model(filename)

# check model structure
model.summary()


# train the model for a few epochs and check if it is improving

# config Guillaume CPU
#EPOCH_0 = 45
#EPOCHS = 10
#histody = train_classifier(model, new_epochs=10, batch_size=16, verbose=1,
#                 initial_epoch=EPOCH_0,
#                 use_checkpoints=True, train_size=2000, test_size=500)

# config Sasha GPU
EPOCH_0 = 45
EPOCHS = 50
train_classifier(model, new_epochs=47, batch_size=16, verbose=1,
                 initial_epoch=EPOCH_0,
                 use_checkpoints=True, train_size=8000, test_size=2000)


#Save model once training is done
model.save("classifier_1_epoch_" + str(EPOCH_0 + EPOCHS) + ".h5")

#TODO Save history for plotting




#Take current model, add dropout, train over 50 epochs and check results


#Take current model, add one intermdiate convolution block
