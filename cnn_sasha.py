"""
Created on Mon May 21 20:55:27 2018
@author: guillaume
New script for Sasha
"""

from cnn_util import train_model #build_model #, print_file_layers


#filename = "classifier_1_epoch_45.h5"
filename = "weights-improvement-19-0.77.h5"
#print_file_layers(filename)


# Load the model from the dead
from keras.models import load_model
model = load_model(filename)

# check model structure
model.summary()


# train the model for a few epochs and check if it is improving
CONFIG = "Speed" # "Guillaume" or "Alexander"

USE_CHECKPOINTS = True
SHOW_LEARN_PARAM = False

if CONFIG == "Guillaume":
    # config Guillaume CPU
    EPOCH_0 = 0
    NEW_EPOCHS = 15
    TRAIN_SIZE = 150
    TEST_SIZE = 200
    BATCH_SIZE = 16
    USE_CHECKPOINTS = False
elif CONFIG == "Speed":
    # config Guillaume CPU
    EPOCH_0 = 0
    NEW_EPOCHS = 15
    TRAIN_SIZE = 10
    TEST_SIZE = 10
    BATCH_SIZE = 16
    USE_CHECKPOINTS = False
elif CONFIG == "Patrick":
    # config Patrick CPU
    EPOCH_0 = 0
    NEW_EPOCHS = 15
    TRAIN_SIZE = 150
    TEST_SIZE = 2000
    BATCH_SIZE = 16
    USE_CHECKPOINTS = False
#classifier.fit_generator(training_set, steps_per_epoch = 150, epochs = 25, validation_data = test_set, validation_steps = 62)    
elif CONFIG == "Alexander":
    # config Sasha GPU
    EPOCH_0 = 45
    NEW_EPOCHS = 50
    TRAIN_SIZE = 8000
    TEST_SIZE = 2000
    BATCH_SIZE = 32 #or 64



history = train_model(model, new_epochs=NEW_EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                 initial_epoch=EPOCH_0,
                 use_checkpoints=USE_CHECKPOINTS, show_learn_param=SHOW_LEARN_PARAM,
                 train_size=TRAIN_SIZE, test_size=TEST_SIZE)


#Save model once training is done
model.save("classifier_1_epoch_" + str(EPOCH_0 + NEW_EPOCHS) + ".h5")

##TODO Save history for plotting
#loss_history = history.history["loss"]
#np_loss_history = np.array(loss_history)
#
#import matplotlib.pyplot as plt
#np_loss_history.shape
#
#plt.plot(np_loss_history)
#
#
#acc_history = history.history["acc"]
#np_acc_history = np.array(acc_history)
#plt.plot(np_acc_history)
#
#
#np.savetxt("loss_history.txt", np_loss_history, delimiter=",")




#Recharge the epoch 19
new_model = load_model("weights-improvement-19-0.77.h5")
#FIXME find a deepcopy from model to model?

#add regularisation on the new model






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
