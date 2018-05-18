# CNN Transfer Learning


The principle is easy: optimize and accelerate the learning of a model by training specific layers,
transfer them to a deeper model and iterate.

This is relevant when the compute capacity gets insufficient compared to the needs.
Cloud GPUS are good but at some point beyond the budget...
This is also a good opportunity to better understand the building and training of models.

## Task list
- [ ] cleanup the readme
- [X] display nice pics of cats/dogs in the readme
- [ ] add a predict function for one picture after the training


## Context
We train a dog and cat classifier over 10000 pictures (5000 dogs, 5000 cats).
8000 are used for training (+ data augmentation) and 2000 for testing.
Directly coming from the [CIFAR10 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

Yes my friends, a big bunch of cute cat and dog pictures such as:

<img src="dataset/training_set/cats/cat.998.jpg" width="100" height="120"><img src="dataset/training_set/cats/cat.2.jpg" width="100" height="120">
<img src="dataset/training_set/dogs/dog.998.jpg" width="100" height="120"><img src="dataset/training_set/dogs/dog.2.jpg" width="100" height="120">

## Data augmentation
10000 pictures is actually very small for computer vision. For that reason, the existing data is multiplicated by a range of techniques:
- mirroring pictures left-right (effectively doubling the dataset size)
- random cropping/zooming
- changing the colors

The data augmentation generator produces variations out of each original picture.
This leads to a wider range of pictures and reduces overfitting.

In practice, with a standard batch_size of 32, the data generator produced 32 new variations out of each original picture.
This lead to 256000 pictures to train the model, and was for my CPU a bit heavy.
It took 25 minutes per epoch (1 pass through the data).
Of course, better results would have most been reached in much less time using a decent GPU.
For that reason, I saved the network parameters weights to conveniently load them again and continue the training later.

That happened to be convenient to transfer weights into new versions of the network.

## Weight transfer
I started with a very small network:

    classifier_1.summary()

    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
    max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
    flatten_1 (Flatten)          (None, 32768)             0         
    dense_1 (Dense)              (None, 128)               4194432   
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 4,195,457
    Trainable params: 4,195,457
    Non-trainable params: 0
    _________________________________________________________________

I trained it and obtained an accuracy rate around 79%.
Fair enough after a few epochs and not so much data.
Then I added a new Convolution2D + MaxPool block and transferred as many weights as possible into the new one.

    classifier_2.summary()

    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_2 (Conv2D)            (None, 64, 64, 32)        896       
    max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
    conv2d_3 (Conv2D)            (None, 32, 32, 32)        9248      
    max_pooling2d_3 (MaxPooling2 (None, 16, 16, 32)        0         
    flatten_2 (Flatten)          (None, 8192)              0         
    dense_3 (Dense)              (None, 128)               1048704   
    dense_4 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 1,058,977
    Trainable params: 1,058,977
    Non-trainable params: 0
    _________________________________________________________________

A full copy from a model to another identical model is actually quite easy.
Saving the weights of a classifier model:

    classifier.save_weights("classifier2_tmp.h5")
    
Loading the weights back into the classifier_2 model

    classifier_2.load_weights("classifier2_tmp.h5")


### Limits of weight transfer
In this case, it worked only with the first convolution layer.
We cannot transfer all the weights between models if they have different structures.
But this can be done partially, layer by layer.

First, a model with the same structure has to be created and weights are loaded into it.

    classifier_1 = Sequential()
    classifier_1.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
    classifier_1.add(MaxPooling2D(pool_size=(2,2)))
    classifier_1.add(Flatten())
    classifier_1.add(Dense(128, activation='relu'))
    classifier_1.add(Dense(1, activation='sigmoid'))

This can also be done more easily by saving and loading the whole model in the h5 file.

    classifier.save("classifier_model_tmp.h5")
    classifier_new.load("classifier_model_tmp.h5")

Then weights we can transferred layer per layer, as long as they have the same type and dimension.

    classifier_2.layers[0].set_weights(classifier_1.layers[0].get_weights())

In this case, only the first layer could be transferred, due to some constaints.
1. Max Pooling has no trainable parameters, so nothing to transfer
1. Flatening is just taking the rank 2 feature matrix out of the convolution+MaxPool layers into a single rank 1 vector.
Also no trainable parameters.
1. After the new convolution+MaxPool in the new model, the feature vector size reduced from 32K to 8K. So it is not possible to load the weights.

At the end, only the first layer could be transferred.
**But it had a positive effect.**
I checked it by fitting the second deeper model over several epochs.
Once with weight transfer, once without.
- After weights transfer, the model reached quickly 82% validation accuracy and stagnated at this level
- without transfer, the model reached quickly 79% validation accuracy and stagnated there

I believe that fitting over many more epochs would have increased the accuracy, but my CPU was not convenient for this.
It was quite interesting to see that training on a very small network and transfer just the first convolution weights could immediately increase the performance of 3%.

### Number of trainable parameters    
The first network, which is quite small, has 4 millions parameters, which is surprisingly high.

This number of trainable parameters of the network, went down to 1 million in the second one!
Intuitively I would have expected more parameters on a deeper network.

In practice, most of the parameters are due to the the flattened vector before the fully connected layer at the end.
This is the feature vector is produced by the Convolution blocks.

The number of parameters between this feature vector and the first fully connected layer is:
vector size x number of nodes (128).

Its size is 32768 in the first network, and 8192 in the second => 4 times less parameters.

The Max pool layer plays here a big role, as each Max Pool layer divides by 4 the number of the resulting information passed to later layers.

Hence, more Convolution2D + Max Pooling layers means less trainable parameters.

### Misc
Transfer learning:
    copy the trained weights from one NN to another
    Here, only the first layer can be transferred as the structures are different
    This is enough to already increase the performance
    
    
    classifier_1 has 4M trainable parameters, classifier_1 only 1M although it is deeper
    Conv2D and MaxPool decrease a lot the features volume so it compresses this amount
