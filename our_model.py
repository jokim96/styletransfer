import numpy as np
from skimage.transform import rescale
import tensorflow as tf
from PIL import Image
import numpy as np
from pylab import imshow, show, get_cmap
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19      
from skimage.exposure import adjust_gamma
from keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Model
from numpy import random
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy import ndimage
import os
import errno
from datetime import datetime

# activate gpu/cpu
 #source /course/cs1430/tf_gpu/bin/activate
 # to clear up memory
 #kill $(jobs -p)

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

def load_data(img_path, target_size=(336,336)): 
        img = image.load_img(img_path, target_size=target_size, interpolation ="bicubic")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

def calc_style_loss(base_style, output):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        gram_style = gram_mat(base_style)
        gram_output = gram_mat(output)
        style_loss =  tf.reduce_mean(tf.square(gram_style - gram_output))
        return style_loss

def calc_content_loss(content, output):
        content_loss = tf.reduce_mean(tf.square(content- output))
        return content_loss

# Initialize gloabal variables
POS_CONTENT, POS_STYLE, POS_COMBINED = (0, 1, 2) 
content_layer = ['block4_conv2'] 
style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_iterations=100 
results = []
#parameters for style and content, later used for saving
style = 'kandinsky'
content = 'galata'
style_path = 'images/' + style + '.jpg'
content_path = 'images/' + content + '.jpg'
i_style = load_data(style_path) 
i_content = load_data(content_path)
img_h = i_content.shape[1]
img_w = i_content.shape[2]
i_combined = K.placeholder((1, img_h, img_w, 3))
 
combined_tensor = K.concatenate([i_content, i_style, i_combined], axis=0)

# Initialize VGG-19 Model using Keras 
model = VGG19(input_tensor=combined_tensor, weights='imagenet', include_top=False)
 
content_weight =  1.0
style_weight= 0.5

def calc_loss(model, combined_image):
        layer_outputs = dict([layer.name, layer.output] for layer in model.layers)
        style_loss = K.variable(0.0); content_loss = K.variable(0.0) 
        # calculate content loss
        for layer in content_layer: 
                content_features = layer_outputs[layer][POS_CONTENT]
                combined_features = layer_outputs[layer][POS_COMBINED]
                content_loss = content_loss + calc_content_loss(content_features, combined_features)

        # calculate style loss
        for layer in style_layer:
                print('layer: ', layer) 
                style_features = layer_outputs[layer][POS_STYLE]
                combined_features = layer_outputs[layer][POS_COMBINED]
                style_loss =  style_loss +calc_style_loss(style_features, combined_features)
        #simpler calc
        total_loss = content_weight * content_loss + style_weight * style_loss 

        return total_loss

total_loss = calc_loss(model, i_combined)
#use gradient
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

# experiment with white noise image
# x = np.random.uniform(0, 255, (1, img_h, img_w, 3)) - 128.

x = x.flatten()
for i in range(num_iterations):
        mf = 20
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                        x,
                                        fprime=evaluator.grads,
                                        maxfun= mf)
        print('Loxx:', min_val)

        #### OUR STUPID MISTAKE HERE #####
        # x = deprocess_image(x, img_h, img_w)
        ##################################
        img = deprocess_image(x, img_h, img_w)
        
        #used a median filter to reduce noise
        img = ndimage.median_filter(img, 3) 

     
        file = 'images/at%d,(%f,%f,%d).png' % (i,content_weight,style_weight, mf)
        end_time = time.time()
        print('This iteration %d completed in %ds' % (i, end_time - start_time))
        filename = "images/%s,%s(%f,%f,%d)/at%d.png" % (content,style,content_weight,style_weight, mf, i)
        #create a new folder for each run style-content pair
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save_img(filename, img)
        print("image saved as", filename)
      