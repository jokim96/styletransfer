
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


    print("created model")


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
        weight_content=1.5
        weight_style=10.0
        weight_denoise=0.3
        num_iterations=120
        step_size=10.0



        content_layer_ids = [4]
        vgg = Vgg16()
        (h,w,d) = content_image.shape
        content_image = np.reshape(content_image, (int(h/30),int(w/30),3))

        #which one?
        model = vgg.build(content_image)
        # model = vgg16.VGG16()

        #parameters
        weight_content=1.5
        weight_style=10.0
        weight_denoise=0.3
        num_iterations=120
        step_size=10.0

        # sess = tf.Session(graph = model)
        session = tf.InteractiveSession(graph=model.graph)
        loss_content = self.calc_content_loss(session=session, model=model, content_image=content_image, layer_ids=content_layer_ids)
        loss_style = self.calc_style_loss(session=session, model=model, content_image=content_image, layer_ids=content_layer_ids)

        adj_content = tf.Variable(1e-10, name='adj_content')
        adj_style = tf.Variable(1e-10, name='adj_style')
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')
        session.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])

        update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
        update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
        update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

        loss_combined = weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

        gradient = tf.gradients(loss_combined, model.input)

        run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

        # The mixed-image is initialized with random noise.
        # It is the same size as the content-image.
        mixed_image = np.random.rand(*content_image.shape) + 128

        for i in range(num_iterations):
            # Create a feed-dict with the mixed-image.
            feed_dict = model.create_feed_dict(image=mixed_image)

            # Use TensorFlow to calculate the value of the
            # gradient, as well as updating the adjustment values.
            grad, adj_content_val, adj_style_val, adj_denoise_val \
            = session.run(run_list, feed_dict=feed_dict)

            # Reduce the dimensionality of the gradient.
            grad = np.squeeze(grad)

            # Scale the step-size according to the gradient-values.
            step_size_scaled = step_size / (np.std(grad) + 1e-8)

            # Update the image by following the gradient.
            mixed_image -= grad * step_size_scaled

            # Ensure the image has valid pixel-values between 0 and 255.
            mixed_image = np.clip(mixed_image, 0.0, 255.0)

            # Print a little progress-indicator.
            print(". ", end="")
                    # Display status once every 10 iterations, and the last.
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Print adjustment weights for loss-functions.
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            # Plot the content-, style- and mixed-images.
            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)

        print()
        print("Final image:")
        plot_image_big(mixed_image)

        # Close the TensorFlow session to release its resources.
        session.close()

        # Return the mixed-image.
        return mixed_image


    # Loss function calculated through mean squared error between the
    # content/style image and output image
    def mean_sqerr(self, tensor_a, tensor_b):
        return tf.reduce_mean(tf.square(tensor_a-tensor_b))

    # use mean_sqerr to calculate the content loss
    def calc_content_loss(self, session, model, c_img, layer_ids):
        layers = model.get_layer_tensors(layer_ids)
        values = session.run(layers, feed_dict=feed_dict)
        with model.graph.as_default():
            layer_losses = []
            for value, layer in zip(values, layers):
                value_const = tf.constant(value)
                loss = self.mean_sqerr(layer, value_const)
                layer_losses.append(loss)
            total_loss = tf.reduce_mean(layer_losses)
        return total_loss

    def calc_style_loss(self, session, model, c_img, layer_ids):
        feed_dict = model.create_feed_dict(image=self.style_image)
        layers = model.get_layer_tensors(layer_ids)
        with model.graph.as_default():
            gram_layers = [self.gram_mat(layer) for layer in layers]
            values = session.run(gram_layers, feed_dict=feed_dict)
            layer_losses = []
            for value, gram_layer in zip(values, gram_layers):
                value_const = tf.constant(value)
                loss = self.mean_sqerr(layer, value_const)
                layer_losses.append(loss)
            total_loss = tf.reduce_mean(layer_losses)
        return total_loss

    # gram matrix for style loss. Multiply matrix by its a transpose
    # Gram matrix is used to calculate loss
    def gram_mat(self, tensor):
        shape = tensor.get_shape()
        num_channels = int(shape[3])
        matrix = tf.reshape(tensor, shape=[-1, num_channels])
        # matrix = tf.reshape(tensor, shape=[-1, int(tensor.get_shape()[3])])
        return tf.matmul(tf.transpose(matrix), matrix)

    def create_denoise_loss(self, model):
        loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
            tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
        return loss
