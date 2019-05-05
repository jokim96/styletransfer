# for testing


from helpers import load_image, save_image, my_imfilter
import numpy as np
from skimage.transform import rescale
import tensorflow as tf
from PIL import Image
import numpy as np

from pylab import imshow, show, get_cmap


from keras import backend as K

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model
from numpy import random


content_layer = 'block4_conv2'
style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

# Load and process img
style_path = 'images/starry-night.jpg'
i_style = image.load_img(style_path, target_size=(224,224))
i_style = image.img_to_array(i_style)
i_style = np.expand_dims(i_style, axis=0)
i_style = preprocess_input(i_style)

content_path = 'images/galata.jpg'
i_content = image.load_img(content_path, target_size=(224,224))
i_content= image.img_to_array(i_content)
i_content = np.expand_dims(i_content, axis=0)
i_content = preprocess_input(i_content)

img_h = i_content.shape[1]
img_w = i_content.shape[2]
 

ph_content = K.variable(i_content)
ph_style = K.variable(i_style)

# Initialize VGG-16 Model using Keras
model = VGG16(weights='imagenet', include_top=False)
content_model = Model(model.input, model.get_layer(content_layer).output)

# pass content image inth the network to extract content feature
content_features = content_model.predict(i_content)

# pass the style image into the network
# iterate through each layer and extract style features
style_features = []
for layer in style_layer:
        style_model = Model(model.input, model.get_layer(layer).output)
        style_features.append(style_model.predict(i_style))
   

def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.
    # Arguments
        x: A numpy-array representing the generated image.
    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Visualization of the layer
# print(features.shape[-1])
for channel in range(style_features[0].shape[-1]): 
    featureMap = style_features[3][:,:,:,channel]
    featureMap = deprocess_image(featureMap)[0]
 
print(i_content.shape)  

#create the white noise image
white_noise = np.random.random([img_w, img_h])
plt.imshow(white_noise, cmap='gray', interpolation='nearest');
plt.show()

ph_whitenoise = K.variable(i_style)

def gram_mat(self, tensor):
        matrix = tf.reshape(tensor, shape=[-1, tensor.get_shape()[3]])
        # matrix = tf.reshape(tensor, shape=[-1, int(tensor.get_shape()[3])])
        return tf.matmul(tf.transpose(matrix), matrix)


def calc_content_loss(content, output):
        return tf.losses.mean_squared_error(content,output)


# A is the style original style representation in the given layer
# G is the style ouput style representation in the given layer
# N is the number of feature maps
# M is the height times the width of the feature map
def calc_style_loss(style, output):
        M = self.img_h * self.img_w
        N = 3   
        A = gram_mat(style)
        G = gram_mat(output)
        style_loss = K.sum(K.square(G-A))/(4*np.power(N,2)*np.power(N,2))
        return style_loss