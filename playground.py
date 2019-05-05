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


def calc_content_loss(content, output):
        return tf.losses.mean_squared_error(content,output)

 
def gram_mat(tensor):
        print(tensor.shape) 
        matrix = tf.reshape(tensor, shape=[-1, tensor.shape[-1]]) 
        return tf.matmul(tf.transpose(matrix), matrix)

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

content_layer = 'block4_conv2'
style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

# Load and process style, content and mixed  img
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

# create the white noise image
output_image = np.random.random([img_w, img_h])
plt.imshow(output_image, cmap='gray', interpolation='nearest')
n_channels = 3
output_image = np.resize(output_image, (img_h, img_w, n_channels))
output_image = image.img_to_array(output_image)
output_image = np.expand_dims(output_image, axis=0)
output_image = preprocess_input(output_image)
 



# Initialize VGG-16 Model using Keras
model = VGG16(weights='imagenet', include_top=False)
content_model = Model(model.input, model.get_layer(content_layer).output)


########################
# EXTRACT FEATURES #####
########################
# pass content image inth the network to extract content feature
content_features = content_model.predict(i_content)

# pass the style image into the network
# iterate through each layer and extract style features
style_features = []
for layer in style_layer:
        style_model = Model(model.input, model.get_layer(layer).output)
        style_features.append(style_model.predict(i_style))
   
# Visualization of the layer
for channel in range(style_features[0].shape[-1]): 
    featureMap = style_features[3][:,:,:,channel]
    featureMap = deprocess_image(featureMap)[0]
 

def get_style_loss(base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.shape 
        gram_style = gram_mat(base_style)
        print(base_style.shape)
        print(gram_target.shape)
        print(gram_style.shape)
        return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

def compute_loss(style_model, content_model, style_layers, loss_weights, mixed_img, gram_mat, content_features, style_features):
        #initialize losses
        style_loss = 0
        content_loss = 0 
        num_style_layers = len(style_features)
        #calulcate losses
        style_weights, content_weights = loss_weights

        # content for the mixed image
        content_output_features = content_model.predict(mixed_img)

        # style for the mixed image
        style_output_features = []
        for layer in style_layer:
                style_model = Model(model.input, model.get_layer(layer).output)
                style_output_features.append(style_model.predict(mixed_img))


        #sum style losses for all layers
        weight_per_style_layer = 1.0 / float(num_style_layers)
        for target_style, comb_style in zip(ma_grams, style_output_features):
                style_loss += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
        
        #sum content losses for all layers
        weight_per_content_layer = 1.0 / float(len(content_features))
        for target_content, comb_content in zip(content_features, content_output_features):
                content_loss += weight_per_content_layer* tf.reduce_mean(tf.square(comb_content[0] - target_content))


        print("style_loss")
        print(content_loss)
        print(content_weights)
        total_loss = style_loss*style_weights + content_loss*content_weights[0]
        return total_loss

#calcualte gram matrices for all style layers in style features
ma_grams =  [gram_mat(tensor) for tensor in style_features]

#initialize optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

#predetermined values
content_weight=1e3
style_weight=1e-2
norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means 

#array that contains the generated images
results = []
num_iterations=1000,
loss_weights = (style_weight, content_weight)

for i in num_iterations:
        loss = compute_loss(style_model, content_model, style_layer, loss_weights, output_image, gram_mat, content_features, style_features)
        grads = K.gradients(loss, output_image)
        optimizer.apply_gradients([(grads, output_image)])
        clipped = tf.clip_by_value(output_image, min_vals, max_vals)
        output_image.assign(clipped)
        if loss < best_loss:
          # Update best loss and best image from total loss. 
                best_loss = loss
                best_img = deprocess_image(output_image.numpy())

        plot_img = output_image.numpy()
        plot_img = deprocess_image(plot_img)
        results.append(plot_img)

for i,img in enumerate(results):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()