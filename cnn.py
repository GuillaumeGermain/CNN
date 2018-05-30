# Convolutional Neural Network


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from cnn_util import print_file_layers, train_model, print_model_layers



# Part 1 - Building the CNN
"""
The models are built in a very clear and explicit way in Keras, layer by layer
There are 2 options:
    The sequential way, where we stack layer on top of each other
        It is simple but allows only 1 stream of layers
    The functional API way, where we define variables for each layer
        This second approach is a bit more complicated but allows much more complicated models
    
    In this "tutorial" we will stick to the sequential way to keep it simple
"""

# First simpler CNN version
classifier_1 = Sequential()
classifier_1.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
classifier_1.add(MaxPooling2D(pool_size=(2,2)))
classifier_1.add(Flatten())
classifier_1.add(Dense(128, activation='relu'))
classifier_1.add(Dense(1, activation='sigmoid'))

# Build a new deeper classifer
classifier_2 = Sequential()
classifier_2.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
classifier_2.add(MaxPooling2D(pool_size=(2,2)))
classifier_2.add(Convolution2D(32, kernel_size=(3,3), padding='same', activation='relu'))
classifier_2.add(MaxPooling2D(pool_size=(2,2)))
classifier_2.add(Flatten())
classifier_2.add(Dense(128, activation='relu'))
classifier_2.add(Dense(1, activation='sigmoid'))


# Part 1 bis - check the model structures 

# Display detailed layer by layer model structure
classifier_1.summary()
classifier_2.summary()

# Another way to see the structure of this model (object-wise)
vars(classifier_1)

# Custom display: layer name and type
print_model_layers(classifier_1)



# Part 2 - Train the model 

# This is what should be done to initially train the first model
# Commented as we will save time and load a pre-trained model
#history_1 = train_model(classifier_1, new_epochs=100)


""" 
The History object contains a list of relevant variables:

the compilation parameters
    validation_data': None,
     'model': <keras.models.Sequential at 0x1853673748>,
     'params': {'epochs': 34,
      'steps': 8000,
      'verbose': 1,
      'do_validation': True,
      'metrics': ['loss', 'acc', 'val_loss', 'val_acc']},
     'epoch': [9,
      10,
    ...
      33],

the "history":
    the list of values corresponding to the metrics visible in the first section
    in this case: ['loss', 'acc', 'val_loss', 'val_acc']
    The most important one is the val_acc, how accurate the model was on the test set, the data that it has not seen before

There are some issues with this history object:
    it is reset every time you further train a model
    for that reason it is better to store it in a variable, at best in a list/dictionary
    it's not the most convenient but you can get back this information
"""


# Part 3 - Save and reload a pre-trained model

"""
Saved weights

    Model 1: 1 convolution block
        sequential_1.h5                 intermediate weights
        sequential_1-epoch-43-0.77.h5   latest weights after longer training
    
    Model 2: 2 convolution block
        classifier2_1.h5                intermediate weights
        sequential_2-epoch-18-0.84.h5   Best performance. trained after ihneriting weights from model 1
    
    Model 3: 3 convolutions blocks
    classifier3.h5
    
    
    Best of the best so far: 90.2%
        sequential_fc3-epoch-12-0.902.h5 with dropout, regularisation, etc.
    
"""


# Reload saved weights back into classifier_1
# This loads only the weights, not the structure
# It works because classifier_1 has exactly the same structure as the one in the file
filename = "models/sequential_1-epoch-43-0.77.h5"
classifier_1.load_weights(filename)




# Optional: Further train classifier 1
"""
train_model function:
    - generates the augmented train and test dataset generators
    - compiles the model
    - trains the model new_epochs times
    - the initial_epoch parameter sets from which we continue the training
      Important as we use an adaptive learning rate, which decreases over iterations

Models can be trained further

"""

# Here we have a model trained over 100 epochs and train it during 20 more epochs
history = train_model(classifier_1, initial_epoch=100, new_epochs=20)


# Demonstration:
# we save the model, delete the model in memory, and reload it again
# Save classifier_1 weights + structure
# This saves the whole model

classifier_1.save("classifier_1_tmp.h5")
# Delete the model in memory
del classifier_1

# Load the model again
from keras.models import load_model
classifier_1 = load_model("classifier_1_tmp.h5")
classifier_1.summary()




# Part 4 - Tranfer learning

# We copy the weights from the first model to to the second one
# It can be done over the whole model if they have EXACTLY the same structure
# classifier_2.set_weights(classifier_1.get_weights())

# Here we copy layer per layer
# This works only from 2 models in memory, and if the layers are of the same type
# Here we copy the Conv2D filter weights to another Conv2D layer

classifier_2.layers[0].set_weights(classifier_1.layers[0].get_weights())


# Then we train the new model further
# The performance starting from scratch is typically very low, starting from 50-60% and increasing
# As we transferred an effective filter from the first Conv2D layer, the initial performance will be already good, and reach faster a plateau
# The accuracy of this second model should be higher as it is deeper.

history_2 = train_model(classifier_2, new_epochs=10)




# plot()


