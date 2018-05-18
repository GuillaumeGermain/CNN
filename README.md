# CNN Transfer Learning


The principle is easy: optimize and accelerate the learning of a model by training specific layers,
transfer them to a deeper model and iterate.

[//]: # (The gradients are fully A kind of bottleneck effect forces the layers to )

This is relevant when the compute capacity gets insufficient compared to the needs.
Cloud GPUS are good but at some point beyond the budget...

This is also a good opportunity to better understand the building and training of models.

## Disclaimer:
**This is still a work in progress and scripts are not finished yet!**

## Context
Training a dog and cat recognizer over 10000 pictures.
8000 are used for training (+ data augmentation) and 2000 for testing.

## Data augmentation technique
The data augmentation, generates variations out of original pictures.
This is done by mirroring pictures left-right (doubling the dataset size), random cropping/zooming and rotating.
This leads to a wider range of pictures and reduces overfitting.

In practice, with a batch_size of 32, the data generator produces 32 new variations out of each original picture.
This leads to 256000 pictures to train the model, which is CPU-wise a bit heavy.
On my CPU, it takes 25 minutes per epoch (1 pass through the data).
For that reason, I save the network parameters weights to load them again and continue the training.

That happens to be convenient to transfer weights to new deeper versions of the network.

## Weight transfer: first lessons learned
I started with a very small network:

"""
classifier_1.summary()
"""

"""
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 32768)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               4194432   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 4,195,457
Trainable params: 4,195,457
Non-trainable params: 0
_________________________________________________________________
"""    

I train it and save the parameter weights.
Then I add a new Convolution2D + MaxPool and transfer as many weights as possible to the new one.
In practice, in this case, it works only with the first convolution layer.

"""
classifier_2.summary()
"""

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 64, 64, 32)        896       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 128)               1048704   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 129       
=================================================================
Total params: 1,058,977
Trainable params: 1,058,977
Non-trainable params: 0
_________________________________________________________________
"""

    
Surprisingly, the number of trainable parameters of the network has been divided by 4 after adding a Convolution + Max Pool block.

I tried



### Misc
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

