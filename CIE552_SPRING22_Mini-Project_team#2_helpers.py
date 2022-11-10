# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from scipy.fft import fft, ifft
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import cv2
import matplotlib.pyplot as plt

def my_imfilter(image: np.ndarray, filter: np.ndarray):
    if (filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0):
        raise NotImplementedError('filter is not defined')
    
    """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
    """

    Im_width=image.shape[1]
    Im_height=image.shape[0]
    filter_width=filter.shape[1]
    filter_height=filter.shape[0]
    if(len(image.shape)==3):
        filtered_image = np.zeros((Im_height, Im_width, image.shape[2]))
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]
        filter = np.pad(filter, ((0, Im_height-filter_height), (0, Im_width-filter_width)), 'constant')

        blue_f=np.fft.fft2(blue)
        green_f = np.fft.fft2(green)
        red_f = np.fft.fft2(red)
        filter_f=np.fft.fft2(filter)
        filtered_image_bf=blue_f*filter_f
        filtered_image_gf = green_f * filter_f
        filtered_image_rf = red_f * filter_f
        filtered_image_b=np.abs(np.fft.ifft2(filtered_image_bf))
        filtered_image_g=np.abs(np.fft.ifft2(filtered_image_gf))
        filtered_image_r=np.abs(np.fft.ifft2(filtered_image_rf))
        filtered_image[:,:,0]=filtered_image_b
        filtered_image[:,:,1]=filtered_image_g
        filtered_image[:,:,2]=filtered_image_r
    elif(len(image.shape)<2):
        filtered_image = np.zeros((Im_height, Im_width))
        filter = np.pad(filter, ((0, Im_height-filter_height), (0, Im_width-filter_width)), 'constant')
        im_f=np.fft.fft2(image)
        filter_f = np.fft.fft2(filter)
        filtered_image_f=im_f*filter_f
        filtered_image=np.abs(np.fft.ifft2(filtered_image_f))
    

    return np.asarray(filtered_image, dtype=np.float32)



def create_gaussian_filter(ksize, sigma):
    x = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    y = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    x_1, y_1 = np.meshgrid(x, y)
    filt = np.exp(-0.5 * (np.square(x_1) + np.square(y_1)) / np.square(sigma))
    filt=filt / np.sum(filt)

    return filt

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel

  
  # Your code here:

  low_frequencies = np.clip(my_imfilter(image1,create_gaussian_filter(31,cutoff_frequency)),0.0,1.0)
  high_frequencies = np.clip(image2 - my_imfilter(image2,create_gaussian_filter(31,cutoff_frequency)),0.0,1.0) # Replace with your implementation
  plt.imshow(low_frequencies)
  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  hybrid_image = np.clip(high_frequencies + low_frequencies,0.0,1.0) # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().

  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!

  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
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
    cur_image = rescale(cur_image, scale_factor, mode='reflect',channel_axis=2)
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
