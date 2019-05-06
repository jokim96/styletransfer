# for testing
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
from scipy.optimize import fmin_l_bfgs_b




def gram_mat(tensor):
        matrix = tf.reshape(tensor, shape=[-1, tensor.shape[-1]])
        return tf.matmul(tf.transpose(matrix), matrix)


def deprocess_image(x):
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

def calc_style_loss(base_style, output):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.shape
        gram_style = gram_mat(base_style)
        gram_output = gram_mat(output)
        style_loss =  tf.reduce_mean(tf.square(gram_style - gram_output))
        print('style_loss ', style_loss)
        return style_loss

def calc_content_loss(content, output):
        return tf.reduce_mean(tf.square(content- output))

def load_data(img_path, target_size=(224,224)):
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

# Initialize macros to keep track
POS_CONTENT = 0
POS_STYLE = 1
POS_COMBINED = 2

content_layer = ['block4_conv2']
style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

# Load and process style, content and mixed  img
style_path = 'images/starry-night.jpg'
content_path = 'images/galata.jpg'
i_style = load_data(style_path)
i_content = load_data(content_path)
img_h = i_content.shape[1]
img_w = i_content.shape[2]
i_combined = K.placeholder((1, img_h, img_w, 3))
# i_style = image.load_img(style_path, target_size=(224,224))
# i_style = image.img_to_array(i_style)
# i_style = np.expand_dims(i_style, axis=0)
# i_style = preprocess_input(i_style)
# i_content = image.load_img(content_path, target_size=(224,224))
# i_content= image.img_to_array(i_content)
# i_content = np.expand_dims(i_content, axis=0)
# i_content = preprocess_input(i_content)

combined_tensor = K.concatenate([i_content, i_style, i_combined], axis=0)
# Initialize VGG-16 Model using Keras
model = VGG16(weights='imagenet', include_top=False, input_tensor=combined_tensor)

######################################
########## EXTRACT FEATURES ##########
######################################
# pass content image inth the network to extract content feature
layer_outputs = dict([layer.name, layer.output] for layer in model.layers)

content_weight=1e3
style_weight=1e-2
style_loss = 0; content_loss = 0

for layer in content_layer:
        content_features = layer_outputs[layer][POS_CONTENT]
        combined_features = layer_outputs[layer][POS_COMBINED]
        content_loss += calc_content_loss(content_features, combined_features)
for layer in style_layer:
        style_features = layer_outputs[layer][POS_STYLE]
        combined_features = layer_outputs[layer][POS_COMBINED]
        style_loss += calc_style_loss(style_features, combined_features)
weight_clayer = 1.0 / float(len(content_layer))
weight_slayer = 1.0 / float(len(style_layer))

total_loss = content_weight* weight_clayer * content_loss + style_weight * weight_slayer * style_loss


########################### OLD #############################
# content_model = Model(model.input, model.get_layer(content_layer).output)
# content_features = content_model.predict(i_content)

# # pass the style image into the network
# # iterate through each layer and extract style features
# style_features = []
# for layer in style_layer:
#         style_model = Model(model.input, model.get_layer(layer).output)
#         style_features.append(style_model.predict(i_style))


# # Visualization of the layer
# for channel in range(style_features[0].shape[-1]):
#     featureMap = style_features[3][:,:,:,channel]
#     featureMap = deprocess_image(featureMap)[0]


#create the white noise image // Are we ever gonna need this?
# output_image = np.random.random([img_w, img_h])
# plt.imshow(output_image, cmap='gray', interpolation='nearest')
# plt.show()
####################################################

#calcualte gram matrices for all style layers in style features
# ma_grams =  [gram_mat(tensor) for tensor in style_features]


gradients = K.gradients(total_loss, i_combined)

#### CHANGE LATER ######
optimized = K.function([i_combined], [total_loss] + gradients)

#predetermined values
norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means

#array that contains the generated images
results = []
num_iterations=1000
for i in range(num_iterations):
        # x, min_val, info = fmin_l_bfgs_b


################# OLD ITERATION #################
# for i in range(1000):
#         output_image = i_content
#         loss = compute_loss(model, loss_weights, output_image, ma_grams, content_features)
#         print('loss is ', loss)
#         print('comination img is ', output_image)

#         grads = K.gradients(loss, output_image)
#         # f_outputs = K.function([output_image], outputs)
#         optimizer.apply_gradients([(grads, output_image)])

#         # optimizer.apply_gradients([(grads, output_image)])
#         clipped = tf.clip_by_value(output_image, min_vals, max_vals)
#         output_image.assign(clipped)
#         if loss < best_loss:
#           # Update best loss and best image from total loss.
#                 best_loss = loss
#                 best_img = deprocess_image(output_image.numpy())

#         plot_img = output_image.numpy()
#         plot_img = deprocess_image(plot_img)
#         results.append(plot_img)
###################################################