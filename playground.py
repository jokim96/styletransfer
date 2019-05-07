# for testing
import numpy as np
from skimage.transform import rescale
import tensorflow as tf
from tensorflow.train import AdamOptimizer
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

def deprocess_image(x, img_h, img_w):
    x = x.copy().reshape((img_h, img_w, 3))  
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_data(img_path, target_size=(224,224)):
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

def calc_style_loss(base_style, output):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.shape
        gram_style = gram_mat(base_style)
        gram_output = gram_mat(output)
        style_loss =  tf.reduce_mean(tf.square(gram_style - gram_output))
        return style_loss

def calc_content_loss(content, output):
        content_loss = tf.reduce_mean(tf.square(content- output))
        return content_loss

# Initialize gloabal variables
POS_CONTENT = 0
POS_STYLE = 1
POS_COMBINED = 2
content_layer = ['block4_conv2']
style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
num_iterations=1000
#array that contains the generated images
results = []

style_path = 'images/starry-night.jpg'
content_path = 'images/galata.jpg'
i_style = load_data(style_path)
i_content = load_data(content_path)
img_h = i_content.shape[1]
img_w = i_content.shape[2]
i_combined = K.placeholder((1, img_h, img_w, 3))

combined_tensor = K.concatenate([i_content, i_style, i_combined], axis=0)

# Initialize VGG-16 Model using Keras
model = VGG16(weights='imagenet', include_top=False, input_tensor = combined_tensor)

# print('ahhhhhh', K.get_value(combined_tensor))
# model = Model(model.input, [layer.output for layer in model.layers])

# opt = AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

# shit that will change in the loop
content_weight = 1e3
style_weight=1e-2


def calc_loss(model, combined_image):
        layer_outputs = dict([layer.name, layer.output] for layer in model.layers)
        style_loss = K.variable(0.0); content_loss = K.variable(0.0)

        # calculate content loss
        for layer in content_layer:
                content_features = layer_outputs[layer][POS_CONTENT]
                combined_features = layer_outputs[layer][POS_COMBINED]
                content_loss += calc_content_loss(content_features, combined_features)

        # calculate style loss
        for layer in style_layer:
                print('layer: ', layer)
                style_features = layer_outputs[layer][POS_STYLE]
                combined_features = layer_outputs[layer][POS_COMBINED]
                style_loss += calc_style_loss(style_features, combined_features)

        # calculate total loss
        weight_clayer = 1.0 / float(len(content_layer))
        weight_slayer = 1.0 / float(len(style_layer))
        total_loss = content_weight* weight_clayer * content_loss + style_weight * weight_slayer * style_loss

        return total_loss

total_loss = calc_loss(model, i_combined)
gradients = K.gradients(total_loss, i_combined)[0]
kfunc = K.function([i_combined], [total_loss, gradients])

class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_h, img_w, 3))
        outs = kfunc([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
evaluator = Evaluator()


min_loss = float('inf')
best_i_combined = None
x = load_data(content_path)

x = x.flatten()
for i in range(num_iterations):

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Loxx:', min_val)
    x = deprocess_image(x, img_h, img_w)
    plt.imshow(x)
    plt.show()
# for i in range(num_iterations):
 
#         # print('total loss ', K.get_value(total_loss))

#         print(gradients)
#         #### CHANGE LATE
#         # opt.apply_gradients([(grad, output_img)])
#         _, val = kfunc([bla])
#         bla -= val * 0.001
#         # felt cute, might delete later x
#         # clipped = tf.clip_by_value(output_img, min_vals, max_vals)
#         # output_img.assign(clipped)

#         plot_img = deprocess_image(bla)
#         plt.imshow(plot_img)
#         plt.show()




# Current ISSUES: 
# 1. There's nothing being fed into the model
# 2. Optimization (K.function, K.gradient) not good right now..... needs to update values


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