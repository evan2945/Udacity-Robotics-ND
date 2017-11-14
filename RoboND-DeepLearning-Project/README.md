[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, a deep neural network was trained to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the techniques applied here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

[//]: # (Image References)

[image1]: ./pics/training_with_hero.png
[image2]: ./pics/training_without_hero.png
[image3]: ./pics/hero_at_distance.png
[image4]: ./pics/behind_target.png
[image5]: ./pics/hero_not_visible.png
[image6]: ./pics/hero_from_afar.png
[image7]: ./pics/model.png
[image8]: ./pics/hyperparameters.png
[image9]: ./pics/Scores.png

## Objective ##
In this project, we are tasked with the building of a fully convolutional network to identify a target and then track that target in an
urban environment. The main steps, outlined below, include building a segmentation network, training the network, and then validating it.

### Fully Convolutional Networks (FCNs)
Why are fully convolutional networks used in this project instead of a traditional convolutional network? In a traditional convolutional network,
you you have multiple convolultional steps, followed by fully connected layers, and then some activation function (softmax). While this works great
for for classification problems (is this a cat?), it doesn't help us if we ask the question "where in this picture is the cat?". This is due to the
convolutional layers being followed by fully connected layers. When the tensors become flattened to feed into the fully connected layers, you
lose all spatial information that was present during the convolution steps. This is where fully convolutional networks shine. We simply replace
the fully connected layers with convolutional layers, and we retain the spatial information that we need for this problem. There are three special steps
that FCNs take advantage of: replace fully connected layers with 1x1 convolutional layers, upsampling through the use of transpose convolutional layers,
and skip connections. These are discussed below.

##### 1x1 Convolutional Layers
As stated above, when convolutional layers are fed into a fully connected layer, the tensors are flattened, destroying any spatial information we had.
1x1 convolutional layers help us avoid this, preserving spatial information. 1x1 convolutional layers help to reduce the dimensionality of the layer, but
gives us the added advantage (over fully connected layers) that during testing our model, we can feed images of any size into
our trained network.

##### Transpose Convolutions
Transpose convolutions allows us to upsample our previous layer to a desired resolution or dimension. The upsampling part of this process
is defined by the stride and padding.

##### Skip Connections
One effect of convolutions is that we narrow down the scope by looking at some features, but lose the bigger picture as a result.
Skip connections helps us to retain a larger view of our problem. This works by connecting the output of one layer to the input of a
non-adjacent layer. This allows the network to use information from multiple resolutions. The network is then able to make more precise
segmentation decisions.

### Semantic Segmentation
Semantic segmentation is the process of assigning meaning to a part of an object. In this project, this is done at the pixel level, assigning each
pixel a class. This helps us derive valuable information about every pixel in an image. This is known as scene understanding. An approach to this
is to use multiple decoders, each of which trains on a separate task (for example, one for segmentation, and another for depth measurement). This allows
a single network to not only predict the class of a pixel, but also how far it is away.

### Building the Network
Now that the background has been explained, the following will go into the building of the network. Keras was the major work horse in this process. We first
begin by building the encoder.


The following functions will be used below
```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

##### Encoder
For our encoder, we are using convolutional layers to reduce down to a 1x1 convolutional layer (instead of a fully connected layer). A technique known
as separable convolutions (or depthwise separable convolutions) was used here. This reduces the number of parameters needed, increasing the efficiency
of the encoder. This is comprised of a convolution performed over each channel of an input layer and is followed by a 1x1 convolution that takes the
output channels from the previous step and then combines them into an output layer. Below is the code to create the encoder for this project:

```python
def encoder_block(input_layer, filters, strides):

    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)

    return output_layer
```

##### Batch Normalization
Batch normalization is based on the idea that we normalize the inputs to layers within the network. During training, we normalize each layer's inputs by using
the mean and variance  of the values in the current mini-batch. Batch normalization gives us a few advantages: networks train faster, allows higher learning rates,
simplifies creation of deeper networks, and provides some regularization. The following code was used to do batch normalization.


##### Decoder
There are several ways to achieve upsampling. In this project, we used bilinear upsampling. This is a resampling technique that utilizes the weighted average
of the four nearest known pixels to estimate  a new pixel intensity value. Bilinear upsampling layers do not contribute as a learnable layer like transposed convolutional
layers, and it can lose some finer details, but it helps to speed up the network. Code for the decoder is below:

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):

    output_layer = bilinear_upsample(small_ip_layer)

    output_layer = layers.concatenate([output_layer, large_ip_layer])

    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    output_layer = separable_conv2d_batchnorm(output_layer, filters)

    return output_layer
```

##### Model
Now that we have our encoder and decoder built, it is time to build the network. For this project, the following code was used:

```python
def fcn_model(inputs, num_classes):

    enc_layer1 = encoder_block(inputs, 64, 2)
    enc_layer2 = encoder_block(enc_layer1, 128, 2)

    one_conv = conv2d_batchnorm(enc_layer2, 256, 1, 1)

    dec_layer1 = decoder_block(one_conv, enc_layer1, 128)
    x = decoder_block(dec_layer1, inputs, 64)

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

As you can see, this isn't a particularly deep model. I was more concerned with keeping everything straightforward, however, it would
be quite easy to extend this to have many more encoding and decoding layers.

##### Hyperparameters
The following hyperparameters were using for training:

![hyperparameters][image8]

#### Predictions
Now we take a look at the results of training. What follows are several images. In each image we have the following (from left to right):
The original image, the ground truth labels, and what the network predicts. Following each image, there is an image giving actual numbers to
 our predictions. Discussion of the results follow below. We have the following predictions:

###### While following the target

![following][image1]

![following_numbers][image4]

###### While the hero is not visible

![not_visible][image2]

![not_visible_numbers][image5]

###### While the hero is in the distance

![from_afar][image3]

![from_afar_numbers][image6]


#### Results Discussion and Improvements

Before we get into the discussion, we should talk about how to measure the performance of the model on the semantic segmentation task.
One popular method is called IoU. This is just the intersection set (an AND operation) over the union set (an OR set). The closer to 1, the
better the model performed in the segmentation task. The results of the above model, using the hyperparameters listed, is
shown in the image below:

![scores][image9]

As you can see, this set of hyperparameters and model has a final grade of 0.43. The situation where this model really struggled is when the hero was at a distance (I had a lot of false negatives).
This can definitely be improved upon. First, a deeper model could be used to get better results. The hyperparameters could be optimized for this task (see next section). Most
importantly, however, is the data could be better. For this project, I used the data provided. While this gave a decent result, this could
be greatly improved by collecting additional data. I spent a decent amount of time collecting my own data, but I had issues with the
network performing worse with the additional data. This definitely could be attributed to me just collecting bad data, not enough
additional data, or some other human error. If I were to try for a better score, I would focus my attention on collecting much more useful data
from the simulation. This was such an amazing project and I thoroughly enjoyed learning these techniques!

### Extra!
So since this was the final project, I decided to have a little extra fun and incorporate something I have been reading about
lately into this project! I decided to implement a simple genetic algorithm in order to help optimize the hyperparameters for this project.
I created another jupyter notebook that borrowed heavily from this project, and used that as a basis to implement the algorithm. That notebook is
named genetic_algorithm.ipynb. If you decide to look at it, I went much more in detail about the algorithm, why I decided to try it out,
why genetic algorithms are used in deep learning, and how I implemented it in this project. I have only run tests so far, as I do not want to use
up my AWS credits before this project is submitted, but after this project is approved, I plan on using the rest of my credits building out full
sets of hyperparameters to run and see what comes back as the optimal hyperparameters. If you get a chance to look at the notebook,
I would love some feedback. Thanks for all of your help throughout this course!
