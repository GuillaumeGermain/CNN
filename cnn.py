# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from cnn_util import print_file_layers, train_model


# First simpler CNN version
classifier_1 = Sequential()
classifier_1.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
classifier_1.add(MaxPooling2D(pool_size=(2,2)))
classifier_1.add(Flatten())
classifier_1.add(Dense(128, activation='relu'))
classifier_1.add(Dense(1, activation='sigmoid'))

classifier_1.layers

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

# Compare model structures
classifier_1.summary()
classifier_2.summary()



"""
Transfer learning:
    copy the trained weights from one NN to another
    Here, only the first layer can be transferred as the structures are different
    This is enough to already increase the performance
    
    Flatten and Max Pool has no trainable parameters then cannot be transferred
    
    we cannot fully transfer as the shapes sizes differ
    So the weight transfer will be only on the first layer
    classifier_1 has 4M trainable parameters, classifier_1 only 1M although it is deeper
    Conv2D and MaxPool decrease a lot the features volume so it compresses this amount
    
    Most of the parameters is the flattened vector size x number of nodes (128) of the first layer.
    classifier_1 feature vector dim is 32768 and classifier_2 8192 -> 4 times less parameters
    
    It's not possible to transfer directly weights from a h5 file to a NN with a different structure
    Then the weights are loaded back on a classifier with exactly the same structure as the saved NN weights
    and the weights are transferred layer by layer from a NN to another
    
    classifier_final.h5 contains the weights of the first classifier after a few epochs
"""


# Saved weights
"""
classifier_1
classifier.h5          intermediate weights
classifier_final.h5    latest weights after longer trainings

classifier_2
classifier2_final.h5   latest weights after training. this one has inherited from the trained weights
classifier2_1.h5       
classifier3.h5
"""
filename = "classifier2_final.h5"
print_file_layers(filename)




# Reload saved weights back into classfier_1
filename = "classifier_final.h5"
classifier_1.load_weights(filename)

# Further train classifier 1
#...


# Save classifier_1 weights
classifier_1.load_weights(filename)
classifier_1.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])


# Save classifier_1 weights + structure
classifier_1.save("classifier_final_all.h5")
del classifier_1

# Load again
from keras.models import load_model
model = load_model("classifier_final_all.h5")
model.summary()

model.save("classifier_1_epoch_45.h5")



# Apply previous weights to the first Conv2D into classifier_2
classifier_2.layers[0].set_weights(classifier_1.layers[0].get_weights())


# classifier2_final.h5 contains weights trained after a few periods from classifier_2
# Apply all weights from 2. NN
classifier_2.load_weights(filename)




history = History()
history = train_model(model, epochs=100, verbose=2)


# acc .9766 val_acc .8240 
# not too bad, high variance
# saved in classifier2_1.h5 -> to rename to final

#X_test_custom = 

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32, #32
                                                 class_mode='binary')

from keras.preprocessing.image import DirectoryIterator
DirectoryIterator.__doc__

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


score = classifier_2.evaluate(X_test, y_test, batch_size=128)

plot()













##TODO
##TF GPU test
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#
##import tensorflow as tf
##sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
#
#
#
#
#import tensorflow as tf
## Creates a graph.
#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
## Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
## Runs the op.
#print(sess.run(c))
#tf.__version__





