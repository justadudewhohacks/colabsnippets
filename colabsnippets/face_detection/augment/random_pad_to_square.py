import random

import numpy as np


def random_pad_to_square(img, abs_boxes, target_size):
  im_h, im_w = img.shape[0:2]

  pad_x = max(0, target_size - im_w)
  pad_y = max(0, target_size - im_h)

  pad_left = random.randint(0, pad_x)
  pad_right = pad_x - pad_left
  pad_top = random.randint(0, pad_y)
  pad_bottom = pad_y - pad_top

  img = np.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='constant')
  abs_boxes = [[pad_left + x, pad_top + y, w, h] for x, y, w, h in abs_boxes]

  return img, abs_boxes
