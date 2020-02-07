import random

import numpy as np


def pad_to_square(img, abs_boxes, abs_landmarks, target_size, mode='random'):
  im_h, im_w = img.shape[0:2]

  pad_x = max(0, target_size - im_w)
  pad_y = max(0, target_size - im_h)

  if mode == 'random':
    pad_left = random.randint(0, pad_x)
    pad_top = random.randint(0, pad_y)
  elif mode == 'center':
    pad_left = int(pad_x / 2)
    pad_top = int(pad_y / 2)
  else:
    raise Exception('pad_to_square - invalid mode: ' + str(mode))

  pad_right = pad_x - pad_left
  pad_bottom = pad_y - pad_top

  img = np.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='constant')
  abs_boxes = [[pad_left + x, pad_top + y, w, h] for x, y, w, h in abs_boxes]
  abs_landmarks = [[(pad_left + x, pad_top + y) for x, y in l] for l in abs_landmarks]

  return img, abs_boxes, abs_landmarks
