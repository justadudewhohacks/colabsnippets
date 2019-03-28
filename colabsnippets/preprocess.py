import cv2
import math
import numpy as np

def resize_preserve_aspect_ratio(img, size):
  height, width = img.shape[:2]
  max_dim = max(height, width)
  ratio = size / float(max_dim)
  shape = (height * ratio, width * ratio)
  resized_img = cv2.resize(img, (int(round(width * ratio)), int(round(height * ratio))))

  return resized_img

def pad_to_square(img):
  if len(img.shape) == 2:
    img = np.expand_dims(img, axis = 2)

  height, width, channels = img.shape
  max_dim = max(height, width)
  square_img = np.zeros([max_dim, max_dim, channels], dtype = img.dtype)

  dx = math.floor(abs(max_dim - width) / 2)
  dy = math.floor(abs(max_dim - height) / 2)
  square_img[dy:dy + height,dx:dx + width] = img

  return square_img