import math
import numpy as np
import tensorflow as tf
from .calculate_iou import calculate_iou

def get_box_grid_position(box, num_cells, anchors):
  x, y, w, h = box
  ct_x = x + (w / 2)
  ct_y = y + (h / 2)
  col = min(num_cells, max(0, math.floor(ct_x * num_cells)))
  row = min(num_cells, max(0, math.floor(ct_y * num_cells)))

  highest_iou = 0
  highest_iou_anchor_idx = 0
  for anchor_idx, anchor in enumerate(anchors):
    aw, ah = anchor
    anchor_box = (0, 0, aw, ah)
    abs_box = (0, 0, w * num_cells, h * num_cells)
    iou = calculate_iou(anchor_box, abs_box)
    if highest_iou < iou:
      highest_iou = iou
      highest_iou_anchor_idx = anchor_idx

  return col, row, highest_iou_anchor_idx
