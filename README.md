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
Let's train a dog and cat classifier over 10000 pictures (5000 dogs, 5000 cats).
8000 are used for training (+ data augmentation) and 2000 for testing.
Pictures from the [CIFAR10 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

Yes my friends, a big bunch of cute cat and dog pictures like this:

<img src="dataset/training_set/cats/cat.998.jpg" width="100" height="120"><img src="dataset/training_set/cats/cat.2.jpg" width="100" height="120">
<img src="dataset/training_set/dogs/dog.998.jpg" width="100" height="120"><img src="dataset/training_set/dogs/dog.2.jpg" width="100" height="120">

## Data augmentation
10000 pictures is actually not a lot for computer vision. For that reason, the existing data is multiplicated by a range of smart techniques:
- mirroring pictures left-right (effectively doubling the dataset size)
- random cropping/zooming
- changing the colors

The data augmentation generator produces variations out of each original picture.
This leads to a wider range of pictures and reduces overfitting.
The test dataset is also augmented with the same technique.

In practice, with a standard batch_size of 32, the data generator produced 32 new variations out of each original picture.
This lead to 256000 pictures to train the model, and was for my CPU a bit heavy.
It took 25 minutes per epoch (1 pass through the data).
For that reason, I saved the network parameters weights to conveniently load them again and continue the training later.

That happened to be convenient to transfer weights into new versions of the network.

## Building the model
Here I build 2 simple models

    classifier_1 = Sequential()
    classifier_1.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
    classifier_1.add(MaxPooling2D(pool_size=(2,2)))
    classifier_1.add(Flatten())
    classifier_1.add(Dense(128, activation='relu'))
    classifier_1.add(Dense(1, activation='sigmoid'))

    classifier_2 = Sequential()
    classifier_2.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
    classifier_2.add(MaxPooling2D(pool_size=(2,2)))
    classifier_2.add(Convolution2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    classifier_2.add(MaxPooling2D(pool_size=(2,2)))
    classifier_2.add(Flatten())
    classifier_2.add(Dense(128, activation='relu'))
    classifier_2.add(Dense(1, activation='sigmoid'))

## Remark about the neural network and trainable parameters    
Who has taken a Computer Vision course or made a few tutorials on the topic, knows that pictures means a lot of processing. 64x64 format is quite small for pictures, and it has already 12228 dimensions. Plug directly a standard neural network, say 1000 nodes, and you have 12M parameters.

That's where convolution networks come handy. They contain in comparison very few parameters and, coupled with a typical max pooling, encode important features from the picture while significantly reducing the size of the resulting feature vector.

We can see this in these 2 summaries:

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

The second network has one more convolution block (conv2d + max pooling)

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


The first network, which is quite small, has already 4 millions parameters.
The second one, although deeper, has 1 million, 4 times less!
Intuitively we would have expected more parameters on a deeper network, wouldn't we?
In practice, each convolution block compresses the data dimension.
The resulting feature dimension is 32768 in the first network, and 8192 in the second => 4 times less parameters.

Most of the parameters are actually between this resulting feature vector and the begining of the standard neural network (dense_n in the table).
The flattening itself transforms the resulting matrix into a flat vector, keeping the same dimention.
The number of parameters between this feature vector and the first fully connected layer is vector dimension x number of nodes (128) + number of biases (128 again).

So overall, when building a CNN, adding more convolutions blocks actually means less trainable parameters.

## Model fitting
Starting with the smaller network, I trained it and obtained a 79% accuracy rate after a few epochs.
Fair enough without a GPU, only a few epochs and not so much data.

Then I added a new Convolution2D + MaxPool block and transferred as many weights as possible into the new one.
A full copy from a model to another identical model is actually quite easy.

Storing the trained weights of a classifier model into a file is useful to reload it later:

    classifier_2.save_weights("classifier2_tmp.h5")
    
Loading these weights again:

    classifier_2.load_weights("classifier2_tmp.h5")

### Limits of weight transfer
We can transfer the whole set of weights from the file only if the encoded model and the new model have the same structure and dimensions.
If not, we can still transfer weight layer by layer.

First, a model with exactly the same structure has to be created and weights are loaded into it.
Either like this:
Or more conveniently by saving and loading the whole model into and from the h5 file.

    classifier.save("classifier_model_tmp.h5")
    classifier_new.load("classifier_model_tmp.h5")

Then weights can then be transferred layer per layer, as long as they have the same type and dimension.

    classifier_2.layers[0].set_weights(classifier_1.layers[0].get_weights())

In this case, only the first layer could be transferred, due to some constaints.
1. Max Pooling has no trainable parameters, so nothing to transfer
1. Flattening just flattens a rank 2 matrix into a flat vector, of course no trainable parameters.
1. After the new convolution block in the new model, the feature vector size reduced from 32K to 8K. Dimensions differ, this layer cannot be transferred
1. It does not make any sense to transfer the last layer as it is based on completely different layers.

At the end, only the first layer could be transferred.
**But it had a positive effect.**
I checked it by fitting the second deeper model over several epochs.
Once with weight transfer, once without.
- After weights transfer, the model reached quickly 82% validation accuracy and stagnated at this level
- without transfer, the model reached quickly 79% validation accuracy and stagnated there

It was quite interesting to see that training on a very small network and transfer just the first convolution weights could immediately increase the performance of 3%.
I believe that pushing this a bit more, training longer the first model before transferring the weight would result in a better accuracy.
Also, fitting the second model over more epochs would have increased the accuracy, but my CPU was not convenient for this.


### Misc
Transfer learning:
    copy the trained weights from one NN to another
    Here, only the first layer can be transferred as the structures are different
    This is enough to already increase the performance
    Of course, better results would have most been reached in much less time using a decent GPU.

    
    classifier_1 has 4M trainable parameters, classifier_1 only 1M although it is deeper
    Conv2D and MaxPool decrease a lot the features volume so it compresses this amount
