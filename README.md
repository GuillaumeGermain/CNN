# CNN Learning Transfer

The principle is easy: optimize and accelerate the learning of a model by training specific layers,
transfer them to a deeper model and iterate.
It is also useful when a model is not deep enough to get a satisfying performance.
You can then build a deeper one, and transfer parameters of the first common layers to partially benefit from the training of the first model.

This is relevant when the compute capacity gets insufficient compared to the needs.
Cloud GPUS are good but at some point beyond the budget...
This is also a good opportunity to better understand the building and training of models.

## Task list
- [X] cleanup the readme
- [X] display nice pics of cats/dogs in the readme
- [X] train these models over a decent number of epochs
- [X] reduce variance with relevant regularisation
- [ ] push a model with 3+ conv blocks over 90% val accuracy
- [ ] add a predict function for one picture after the training
- [ ] display last used learning rate

## Context
Let's train a dog and cat classifier over 10000 pictures (5000 dogs, 5000 cats).
8000 are used for training (+ data augmentation) and 2000 for testing.
Pictures from the [CIFAR10 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

Yes my friends, a big bunch of cute cat and dog pictures like this:

<img src="dataset/training_set/cats/cat.998.jpg" width="100" height="120"><img src="dataset/training_set/cats/cat.2.jpg" width="100" height="120">
<img src="dataset/training_set/dogs/dog.998.jpg" width="100" height="120"><img src="dataset/training_set/dogs/dog.2.jpg" width="100" height="120">

## Data augmentation
10000 pictures is actually not a lot for a computer vision task. That's why we multiplicate the existing data with a range of simple techniques:
- mirroring pictures left-right (effectively doubling the dataset size)
- random cropping/zooming
- adjusting the colors

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
                                                batch_size=batch_size
                                                class_mode='binary')
    history = model.fit_generator(training_set,
                         steps_per_epoch=train_size,
                         epochs=initial_epoch + new_epochs,
                         validation_data=test_set,
                         validation_steps=test_size,
                         initial_epoch=initial_epoch,
                         verbose=verbose,
                         callbacks=callback_list)

This data augmentation generator produces small variations out of each original picture, and it's even done on the fly, for a best efficiency.
This leads to a much wider range of pictures and helps reducing overfitting.
The test dataset is also augmented using the same technique.

In practice, with a typical batch_size of 32, the data generator produces new variations out of each original picture.
That means, from our original set of 8000 pictures for the training set, we actually get 256000, which is clearly better for a proper training.
In my case, I use a batch_size of 16, and it still takes around 25 minutes per epoch (1 pass through the data).
For that reason, I saved the network parameters weights to conveniently load them again and continue the training later.

That happened to be convenient to transfer weights into new versions of the network.

## Building the model
Let's start by 2 simple models

    # Model 1
    classifier_1 = Sequential()
    classifier_1.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
    classifier_1.add(MaxPooling2D(pool_size=(2,2)))
    classifier_1.add(Flatten())
    classifier_1.add(Dense(128, activation='relu'))
    classifier_1.add(Dense(1, activation='sigmoid'))

    # Model 2
    classifier_2 = Sequential()
    classifier_2.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=(64,64,3), activation='relu'))
    classifier_2.add(MaxPooling2D(pool_size=(2,2)))
    classifier_2.add(Convolution2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    classifier_2.add(MaxPooling2D(pool_size=(2,2)))
    classifier_2.add(Flatten())
    classifier_2.add(Dense(128, activation='relu'))
    classifier_2.add(Dense(1, activation='sigmoid'))

Storing the trained weights of a classifier model into a file is useful to reload it later:

    filename = "classifier1.h5"
    classifier_1.save_weights(filename)
    
ReLoad these weights later back into the model:

    from keras.models import load_model
    classifier_1 = load_model(filename)

## Remark about the neural network and trainable parameters    
Who has taken a Computer Vision course or made a few tutorials on the topic, knows that pictures requires a lot of compute. 64x64 format is quite small for pictures, and it has already 12228 dimensions. Plug directly a standard neural network on that, with say 1000 nodes, and you already have 12M parameters. This would moreover output results based on exact pixel positions, which is really not desirable.

That's where convolution networks come handy. They contain in comparison very few parameters and, coupled with a typical max pooling, they encode important features from the picture while significantly reducing the size of the resulting feature vector.

Having a look again on the network structures:

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


The first network, though quite shallow, has already 4 millions parameters.
The second deeper one has 1 million, 4 times less!
Intuitively we would have expected more parameters on a deeper network, wouldn't we?

Let's get why: both models have a convolution part, followed by a flattening to get a feature vector which is plugged into a standard neural network.
Most parameters are actually between this resulting feature vector and the begining of the standard neural network (dense_3 and dense_4 in tis table).
The number of parameters between the feature vector and the first NN layer is vector dim (32768 or 8192) x number of nodes (128) + number of biases (128 again).

Each convolution block actually compresses the dimension of this feature vector, by a factor 4.

So overall, while building a CNN, adding more convolutions blocks actually means less trainable parameters.

## Model fitting and transfer
Starting with the smaller network, I trained it and obtained a 79% accuracy rate after a few epochs.
Fair enough without a GPU, only a few epochs and not so much data.

Then I added a new Convolution2D + MaxPool block and transferred as many weights as possible into the new one.
A full copy from a model to another identical model is actually quite easy.

### Limits of weight transfer
I quickly noticed that we cannnot transfer weights between models with different structures.
The transfer actually works layer by layer, and only if the layer type and dimensions correspond.
You cannot transfer a Dense layer to a Maxconvolution layer for instance.
Some layers also happen not to have parameters, like MaxPool and Flatten.

So basically, you start with the first layer, transfer parameters, and continue layer by layer until you meet a different type of layer.

    model_new.layers[0].set_weights(classifier_1.layers[0].get_weights())

In the current example, only the first layer can be transferred.
The 2nd is a Max Pool where there is nothing to transfer.
The 3rd layers are already different (Conv2D vs flatten)

Then in prctice, I transferred parameters layers by layer, and only the first ones.

Then weights can then be transferred layer per layer, as long as they have the same type and dimension.
In this case, only the first layer could be transferred, due to some limits.
- Max Pooling has no trainable parameters, so nothing to transfer
- Flattening just flattens a rank 2 matrix into a flat vector, of course no trainable parameters.
- After the new convolution block in the new model, the feature vector size reduced from 32K to 8K. Dimensions differ, this layer cannot be transferred
- It does not make any sense to transfer the last layer as it is based on completely different layers.

At the end, only the first layer could be transferred, **but it had a positive effect.**

I checked it by fitting the second deeper model over several epochs.
Once with weight transfer, once without.
- After weights transfer, the model reached quickly 82% validation accuracy and stagnated at this level
- without transfer, the model reached quickly 79% validation accuracy and stagnated there

## Conclusion
It's quite interesting to see that training a very small network and just transferring its first convolution weights immediately increases the performance of 3% of the second network. And this advantage lasts over at least a few training epochs.
I believe that this small advantage also transfers on bigger models and can save quite some computing time.
Pushing this a bit more, training longer the first model before transferring the weights could result in a better accuracy.

# Further progress
I trained a few models and of course used the recommended Adam optimizer. It speeds up the convergence at first and then slows down later not to overshoot while oscillating around an optimum.

I wanted to check the value of the learning rate over the last epoch. I got actually crazy looking for this simple information.
It is surpringly hard to find. Serious, I did not find it. Most methods I found returned only the INITIAL learning rate, you know, before decay and momentum. Or one very smart method could return it, because it was manually modified and easy to provide.
The only reliable way, at the end, is to manually calculate the Adam learning rate based on the formula, iteration number, decay rate and beta values. OK...

Well, looking for this futile information, I have been digging into any possible Keras documentation, stackoverflow and online forums, the Keras object model, layer and optimizer variables.
I could only get the INITIAL learning rate, before dynamic changes.
That was definitively useful.

### Interesting findings:

#### Model structure
A simple way to dig into the model object:

    vars(model)
    vars(model.optimizer)
    vars(model.layers)

This simple approach enables to understand the internal structure of the Keras object model and look where specific data is located.
The initial learning rate is for instance located **here**:

    model.optimizer.lr

#### Saving models as files every time the validation accuracy has increased
This one is my holy grail. Really useful.
How to train models days and nights on my petty CPU and it automatically saves the best models without fearing power outages or bad manipulations/keystrokes.

    filepath = model.name + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
    callback_list.append(checkpoint)
    
    history = model.fit_generator(training_set,
                         steps_per_epoch=train_size,
                         epochs=initial_epoch + new_epochs,
                         validation_data=test_set,
                         validation_steps=test_size,
                         initial_epoch=initial_epoch,
                         callbacks=callback_list)

Note that the size of files depends on the number of trainable parameters.
Paradoxally, the more convolution blocks, the smaller the number of parameters, and the smaller the files. Yes!

#### Reduce LR on plateau
This one is also very useful.
It triggers a 80% reduction of the learning rate once a plateau has been reached, after a default "patience" of 5 epochs.
According to the Keras documentation and a few blogs, most models benefit from it. The obvious reason is that at some point, the model dances around a local minimum and always arrives beyond, as it jumps to far, due to a too high learning rate.
Reducing 80% this learning rate ensure smaller steps and enables getting closer to it.

    callback_list = []
    
    # Reduce LR on plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=5, min_lr=1.e-5)
    callback_list.append(reduce_lr)
    model.compile('adam', loss='binary_crossentropy', metrics=metric_list)

    history = model.fit_generator(training_set,
                         steps_per_epoch=train_size,
                         epochs=initial_epoch + new_epochs,
                         validation_data=test_set,
                         validation_steps=test_size,
                         initial_epoch=initial_epoch,
                         callbacks=callback_list)

So damn simple...

