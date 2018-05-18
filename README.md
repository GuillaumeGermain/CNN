# CNN Transfer Learning


The principle is easy: optimize and accelerate the learning of a model by training specific layers,
transfer them to a deeper model and iterate.

[//]: # (The gradients are fully A kind of bottleneck effect forces the layers to )

This is relevant when the compute capacity gets insufficient compared to the needs.
Cloud GPUS are good but at some point beyond the budget...

This is also a good opportunity to better understand the building and training of models.

## Disclaimer:
:construction: This is still a work in progress and scripts are not finished yet!

## Task list
- [] cleanup the readme
- [] display nice pics of cats/dogs in the readme
- [] add a predict function for one picture after the training


## Context
We train a dog and cat classifier over 10000 pictures (5000 dogs, 5000 cats).
8000 are used for training (+ data augmentation) and 2000 for testing.

## Data augmentation technique
The data augmentation, generates variations out of original pictures.
This is done by mirroring pictures left-right (doubling the dataset size), random cropping/zooming and rotating.
This leads to a wider range of pictures and reduces overfitting.

In practice, with a batch_size of 32, the data generator produces 32 new variations out of each original picture.
This leads to 256000 pictures to train the model, which is CPU-wise a bit heavy.
On my CPU, it takes 25 minutes per epoch (1 pass through the data).
For that reason, I save the network parameters weights to load them again and continue the training.

That happened to be convenient to transfer weights to new deeper versions of the network.

## Weight transfer: first lessons learned
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

I trained it and saved the parameter weights.

The accuracy rate was around 79%

Then I added a new Convolution2D + MaxPool and transferred as many weights as possible to the new one.
In practice, in this case, it worked only with the first convolution layer.

    classifier_2.summary()

    _________________________________________________________________
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

## Transferring the weights
A full copy from a model to another identical model is quite easy.

Saving the weights of a model:

    classifier.save_weights("classifier2_tmp.h5")
    
    # Reload saved weights back into classifier_2
    classifier_2.load_weights("classifier2_tmp.h5")

This loads all the weights stored in the classifier2_final.h5 file into the classifier_2 model.

### Limits of weight transfer
We actually cannot transfer all the weights between models with different structures.
But it can be done partially, layer by layer.

First a model with the same structure has to be created and weights are loaded into it.
Then we can transfer relevant weights layer per layer, as long as they have the same type and dimension.

In this case, I could only transfer the first layer.
- Max Pooling has no trainable parameters, then nothing to transfer
- Flatening is just taking the rank 2 feature matrix out of the convolution+MaxPool layers into a single rank 1 vector.
Also no trainable parameters.
- After the new convolution+MaxPool in the new model, the feature vector size reduced from 32K to 8K. So it is not possible to load the weights.

At the end, only the first layer could be transferred.
**But it had a positive effect!**

I ran several epochs of the second deeper model with and without training effect to compare.
- After weights transfer, the model reached quickly 82% validation accuracy and stagnated there
- without transfer, the model reached quickly 82% validation accuracy and stagnated there

I believe that fitting over many more epochs would have increased the accuracy, but my CPU was not convenient for this.
It was quite interesting to see that training on a very small network and transfer just the first convolution weights could immediately increase the performance of 3%.

### Number of parameters    
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
    
   
    It's not possible to transfer directly weights from a h5 file to a NN with a different structure
    Then the weights are loaded back on a classifier with exactly the same structure as the saved NN weights
    and the weights are transferred layer by layer from a NN to another
    
    classifier_final.h5 contains the weights of the first classifier after a few epochs

, encoding information what the network could 
