# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt 


def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c)
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message. 
  """ 
  # Get the filter and image dimensions
  (k,l) = (filter.shape)
  (m,n,c) = (image.shape) 

  filtered_image = np.zeros((m,n,c))  
  if (k % 2) == 0 or (l % 2) == 0: 
    raise Exception('my_imfilter function only accepts filters with odd dimensions') 
   
  #  calculate offset for padding
  offsetm = ((k-1)//2)
  offsetn = ((l-1)//2)
  npad = (offsetm, offsetm), (offsetn, offsetn), (0,0) 
  
  paddedimage = np.pad(image, (npad), 'reflect') 
  # loop through all of the dimensions and filter the image
  for n1 in range (n):
    for m1 in range(m):
      for c1 in range (c):
        filtered_image[m1,n1,c1] = (filter*paddedimage[m1:m1+k, n1:n1+l, c1]).sum()  

  return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies. 
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  s, k = cutoff_frequency, cutoff_frequency*2
  probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
  kernel = np.outer(probs, probs)

  #generate the low_frequency image
  low_frequencies = my_imfilter(image1, kernel)  
  #generate the high frequency image and clip the values
  high_frequencies = image2-my_imfilter(image2, kernel)  
  high_frequencies = np.clip(high_frequencies+0.5, 0.0, 1.0)
  # Combine the high frequencies and low frequencies
  hybrid_image = low_frequencies + high_frequencies  

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  hybrid_image = np.clip(hybrid_image-0.5, 0.0, 1.0) 
  
  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect')
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
