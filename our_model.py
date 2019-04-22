
# from tensorpack import *
# from tensorpack.tfutils.symbolic_functions import *
# from tensorpack.tfutils.summary import *
# from tensorpack.tfutils.tower import get_current_tower_context
# from tensorflow.python.training import optimizer
import tensorflow as tf
import hyperparameters as hp
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import rescale
from helpers import load_image, save_image, my_imfilter
from PIL import Image
import cv2
 
print("in OurModel")
class OurModel():
    
    def __init__(self):
        super(OurModel, self).__init__()
        self.use_bias = True 

    #to be used for tensorflow
    # def _get_inputs(self):
    #     return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
    #             InputDesc(tf.int32, [None], 'label')]
 
    #     plot_images(content_image=content_image,
    #                 style_image=style_image,
    #                 mixed_image=mixed_image)
            
    print("created model") 

    def load_data(self):
    #datadir, task, train_or_test
        filename = 'images/bosphorus.jpg'
        content_image = load_image('images/bosphorus.jpg')
        content_image = rescale(content_image, 0.7, mode='reflect')  
        filename = 'images/starry-night.jpg'
        style_image = load_image('images/starry-night.jpg')
        # style_image = cv2.resize(style_image, test_image.shape, interpolation = cv2.INTER_AREA)
        self.orgstyle_image = style_image
        style_image = np.resize(style_image, (content_image.shape))
        # style_image = np.float32(style_image)
        self.content_image = content_image
        self.style_image = style_image
        # self.data = [content_image, style_image] 


    def style_transfer(self):
        #insert algorithm here
        mixed_image = 0.5 * self.content_image + 0.5 * self.style_image

        # mixed_image = np.divide(self.content_image+self.style_image,2)
        plt.figure()
        plt.imshow(self.orgstyle_image)
        plt.show()
        plt.imshow(self.content_image)
        plt.show()
        plt.imshow(mixed_image)
        plt.show()

 
 