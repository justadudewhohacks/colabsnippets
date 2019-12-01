import random

import cv2

from colabsnippets.utils import filter_abs_boxes


def anchor_based_sampling(img, abs_boxes, target_box_anchors, max_scale=4.0, target_range=[1.0, 1.2],
                          min_target_box_px_size=8):
  target_box_width = random.choice(filter_abs_boxes(abs_boxes, min_target_box_px_size))[2]
  filtered_target_box_anchors = [aw for aw in target_box_anchors if aw <= (target_box_width * max_scale)]
  if len(filtered_target_box_anchors) < 1:
    return img, abs_boxes

  target_size = random.choice(filtered_target_box_anchors) * random.uniform(target_range[0], target_range[1])
  scale = target_size / target_box_width
  img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
  abs_boxes = [[int(v * scale) for v in abs_box] for abs_box in abs_boxes]
  return img, abs_boxes
