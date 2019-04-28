
# from tensorpack import *
# from tensorpack.tfutils.symbolic_functions import *
# from tensorpack.tfutils.summary import *
# from tensorpack.tfutils.tower import get_current_tower_context
# from tensorflow.python.training import optimizer
from vgg16 import Vgg16
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

    # def load_data(self):
    # #datadir, task, train_or_test
    #     filename = 'images/bosphorus.jpg'
    #     content_image = load_image('images/bosphorus.jpg')
    #     content_image = rescale(content_image, 0.7, mode='reflect')  
    #     filename = 'images/starry-night.jpg'
    #     style_image = load_image('images/starry-night.jpg')
    #     # style_image = cv2.resize(style_image, test_image.shape, interpolation = cv2.INTER_AREA)
    #     self.orgstyle_image = style_image
    #     style_image = np.resize(style_image, (content_image.shape))
    #     # style_image = np.float32(style_image)
    #     self.content_image = content_image
    #     self.style_image = style_image
    #     # self.data = [content_image, style_image] 

    # def _build_graph(self, inputs):
    #     image, label = inputs
    #     with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
    #         logits = (LinearWrap(image)

    #                     .Conv2D('conv1_1', 64)
    #                     .Conv2D('conv1_2', 64)
    #                     .MaxPooling('pool1', 2)
    #                     # 112
    #                     .Conv2D('conv2_1', 128)
    #                     .Conv2D('conv2_2', 128)
    #                     .MaxPooling('pool2', 2)
    #                     # 56
    #                     .Conv2D('conv3_1', 256)
    #                     .Conv2D('conv3_2', 256)
    #                     .Conv2D('conv3_3', 256)
    #                     .MaxPooling('pool3', 2)
    #                     # 28
    #                     .Conv2D('conv4_1', 512)
    #                     .Conv2D('conv4_2', 512)
    #                     .Conv2D('conv4_3', 512)
    #                     .MaxPooling('pool4', 2)
    #                     .tf.stop_gradient()
    #                     # 14
    #                     .Conv2D('conv5_1', 512)
    #                     .Conv2D('conv5_2', 512)
    #                     .Conv2D('conv5_3', 512)
                        
    #                     .MaxPooling('pool5', 2)
    #                     .Dropout()
    #                     # 7
    #                     .Dropout()
    #                     .FullyConnected('fc6', 4096, nl=tf.nn.relu)
    #                     .Dropout()
    #                     .FullyConnected('fc7', 4096, nl=tf.nn.relu)
    #                     .Dropout()
    #                     .FullyConnected('fc8', out_dim=15, nl=tf.identity)()
    #                     )

    #         prob = tf.nn.softmax(logits, name='output')

    #         cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    #         cost = tf.reduce_mean(cost, name='cross_entropy_loss')

    #         wrong = prediction_incorrect(logits, label)

    #         # monitor training error
    #         add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

    #         add_moving_summary(cost)

    #         add_param_summary(('.*/W', ['histogram']))   # monitor W
    #         self.cost = tf.add_n([cost], name='cost')


    def style_transfer(self, data):
        content_image = data[0]
        orgstyle_image = data[0]
        style_image = data[1]

        vgg = Vgg16()
        model = vgg.build()
        sess = tf.Session(graph = model)


        print("data is received.")
        # mixed_image = np.divide(self.content_image+self.style_image,2)
        plt.figure()
        plt.imshow(self.orgstyle_image)
        plt.show()
        plt.imshow(self.content_image)
        plt.show()
        plt.imshow(self.mixed_image)
        plt.show()


    # Loss function calculated through mean squared error between the 
    # content/style image and output image 
    def mean_sqerr(tensor_a, tensor_b):
        return tf.reduce_mean(tf.square(tensor_a-tensor_b))

    # use mean_sqerr to calculate the content loss
    def calc_content_loss(sess, model, c_img, layer_ids):
        layers = model.get_layer_tensors(layer_ids)

    # gram matrix for style loss. Multiply matrix by its a transpose
    # Gram matrix is used to calculate loss
    def gram_mat(tensor):
        matrix = tf.reshape(tensor, shape=[-1, int(tensor.get_shape()[3])])
        
        return tf.matmul(tf.transpose(matrix), matrix)
