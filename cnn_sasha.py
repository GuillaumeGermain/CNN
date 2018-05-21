"""
Created on Mon May 21 20:55:27 2018
@author: guillaume
New script for Sasha
"""

import numpy as np
from cnn_util import train_classifier #, print_file_layers


filename = "classifier_1_epoch_45.h5"
#print_file_layers(filename)


# Load the model from the dead
from keras.models import load_model
model = load_model(filename)

# check model structure
model.summary()


# train the model for a few epochs and check if it is improving
CONFIG = "GPU"


if CONFIG == "CPU":
    # config Guillaume CPU
    EPOCH_0 = 45
    EPOCHS = 10
    TRAIN_SIZE = 2000
    TEST_SIZE = 500
    BATCH_SIZE = 16
else:
    # config Sasha GPU
    EPOCH_0 = 45
    EPOCHS = 50
    TRAIN_SIZE = 8000
    TEST_SIZE = 2000
    BATCH_SIZE = 32 #or 64


history = train_classifier(model, new_epochs=10, batch_size=BATCH_SIZE, verbose=1,
                 initial_epoch=EPOCH_0,
                 use_checkpoints=True, train_size=TRAIN_SIZE, test_size=TEST_SIZE)


#Save model once training is done
model.save("classifier_1_epoch_" + str(EPOCH_0 + EPOCHS) + ".h5")

#TODO Save history for plotting
loss_history = history.history["loss", "acc"]
np_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", np_loss_history, delimiter=",")





# Further CNN extension

# Importing the Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Convolution2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense


#Take current model, add dropout, train over 50 epochs and check results
#model_2 = model

#Take current model, add one intermdiate convolution block
