import numpy as np
import argparse
import os
import cv2
import sys
from glob import glob
from our_model import OurModel
import hyperparameters as hp
from skimage.transform import rescale
import PIL.Image
from helpers import load_image, save_image, my_imfilter

#to be imported whne tensorflow is implemented  
# from tensorpack import *
# from tensorpack.tfutils.sessinit import get_model_loader
# from tensorpack.tfutils.symbolic_functions import *
# from tensorpack.tfutils.summary import *
# from tensorpack.utils.gpu import get_nr_gpu
# from tensorpack.dataflow.base import RNGDataFlow
# from vgg_model import VGGModel


def load_data(): 
    content_image = load_image('images/galata.jpg')
    content_image = rescale(content_image, 0.7, mode='reflect')   
    style_image = load_image('images/starry-night.jpg')
    style_image = np.resize(style_image, (content_image.shape))
    data = [content_image, style_image] 
    return data
 
 

"""
Program argument parsing, data setup, and training
"""
if __name__ == '__main__':
    data = load_data()
    

    model = OurModel()
    # model.load_data()
    model.style_transfer(data)
    # model.style_transfer(data[0], data[1])

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--task',
    #     required=True,
    #     choices=['1', '2'],
    #     help='Which task of the assignment to run - training from scratch (1), or fine tuning VGG-16 (2).')
    # # Set GPU to -1 to not use a GPU.
    # parser.add_argument('--gpu', help='Comma-separated list of GPU(s) to use.')
    # parser.add_argument(
    #     '--load',
    #     # Location of pre-trained model
    #     # - As a relative path to the student distribution
    #     default='vgg16.npy',
    #     # - As an absolute path to the location on the Brown CS filesystem
    #     #default='/course/cs1430/pretrained_weights/vgg16.npy',
    #     help='Load VGG-16 model.')
    # parser.add_argument(
    #     '--data',
    #     # Location of 15 Scenes dataset
    #     # - As a relative path to the student distribution
    #     default=os.getcwd() + '/../data/',
    #     # - As an absolute path to the location on the Brown CS filesystem
    #     #default='/course/cs1430/datasets/15SceneData/',
    #     help='Location where the dataset is stored.')

    # args = parser.parse_args()

    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # logger.auto_set_dir()

    # dataset_train = get_data(args.data, args.task, 'train')
    # dataset_test = get_data(args.data, args.task, 'test')
 