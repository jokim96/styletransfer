import numpy as np
import argparse
import os
import cv2
import sys
from glob import glob
from our_model import OurModel
import hyperparameters as hp
import PIL.Image

#to be imported whne tensorflow is implemented  
# from tensorpack import *
# from tensorpack.tfutils.sessinit import get_model_loader
# from tensorpack.tfutils.symbolic_functions import *
# from tensorpack.tfutils.summary import *
# from tensorpack.utils.gpu import get_nr_gpu
# from tensorpack.dataflow.base import RNGDataFlow
# from vgg_model import VGGModel


def load_data():
#datadir, task, train_or_test
    content_filename = 'images/bosphorus.jpg'
    content_image = load_image(content_filename, max_size=None)
    content_filename = 'images/starry-night.jpg'
    style_image = load_image(content_filename, max_size=None)
    data = [content_image, style_image] 
    return data

def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)
    
        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

"""
Program argument parsing, data setup, and training
"""
if __name__ == '__main__':
    # data = load_data()
    

    model = OurModel()
    model.load_data()
    model.style_transfer()
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

    # # TensorPack: Training configuration
    # config = TrainConfig(
    #     model=OurModel() if args.task == '1' else VGGModel(),
    #     dataflow=dataset_train,
    #     callbacks=[
    #         # Callbacks are performed at the end of every epoch.
    #         #
    #         # For instance, we can save the current model
    #         ModelSaver(),
    #         # Evaluate the current model and print out the loss
    #         InferenceRunner(dataset_test,
    #                         [ScalarStats('cost'), ClassificationError()])
    #         #
    #         # You can put other callbacks here to change hyperparameters,
    #         # etc...
    #         #
    #     ],
    #     max_epoch=hp.num_epochs,
    #     nr_tower=max(get_nr_gpu(), 1),
    #     session_init=None if args.task == '1' else get_model_loader(args.load)
    # )
    # # TensorPack: Training with simple one at a time feed into batches
    # launch_train_with_config(config, SimpleTrainer())