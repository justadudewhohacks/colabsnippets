import math
import numpy as np
import tensorflow as tf
from .calculate_iou import calculate_iou

def in_grid_range(val, num_cells):
  return min(num_cells - 1, max(0, val))

def get_box_grid_position(box, num_cells, anchors):
  x, y, w, h = box
  ct_x = x + (w / 2)
  ct_y = y + (h / 2)
  col = in_grid_range(math.floor(ct_x * num_cells), num_cells)
  row = in_grid_range(math.floor(ct_y * num_cells), num_cells)

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

def create_gt_mask(batch_gt_boxes, num_cells, anchors):
  batch_size = len(batch_gt_boxes)
  mask = np.zeros([batch_size, num_cells, num_cells, len(anchors), 1])
  for batch_idx in range(0, batch_size):
    for gt_box in batch_gt_boxes[batch_idx]:
      col, row, anchor_idx = get_box_grid_position(gt_box, num_cells, anchors)
      mask[batch_idx, col, row, anchor_idx, :] = 1

  return mask

def create_gt_coords(batch_gt_boxes, num_cells, anchors):
  batch_size = len(batch_gt_boxes)
  gt_coords = np.zeros([batch_size, num_cells, num_cells, len(anchors), 4])
  for batch_idx in range(0, batch_size):
    for gt_box in batch_gt_boxes[batch_idx]:
      col, row, anchor_idx = get_box_grid_position(gt_box, num_cells, anchors)

      x, y, w, h = gt_box
      aw, ah = anchors[anchor_idx]
      gt_x = in_grid_range((x * num_cells), num_cells) - col
      gt_y = in_grid_range((y * num_cells), num_cells) - row
      gt_w = math.log((w * num_cells) / aw)
      gt_h = math.log((h * num_cells) / ah)
      gt_coords[batch_idx, col, row, anchor_idx, :] = [gt_x, gt_y, gt_w, gt_h]

  return gt_coords

def extract_centers_scales_and_scores(pred):
  num_anchors = pred.shape.as_list()[3] / 5
  get_shape = lambda size: np.concatenate((pred.shape.as_list()[0:3], [num_anchors, size]), axis = None)
  grid_preds = tf.reshape(pred, get_shape(5))
  grid_pred_coords = tf.slice(grid_preds, [0, 0, 0, 0, 0], get_shape(4))
  grid_pred_scores = tf.slice(grid_preds, [0, 0, 0, 0, 4], get_shape(1))
  return grid_pred_coords, grid_pred_scores

def compile_loss_op(pred, gt_coords, mask, coord_scale = 1.0, object_scale = 5.0, no_object_scale = 1.0):
  grid_pred_coords, grid_pred_scores = extract_centers_scales_and_scores(pred)
  # TODO: ious
  ious = 1
  object_loss = object_scale * tf.reduce_sum(mask * (ious - tf.nn.sigmoid(grid_pred_scores))**2)
  coord_loss = coord_scale * tf.reduce_sum(mask * (grid_pred_coords - gt_coords)**2)
  no_object_loss = no_object_scale * tf.reduce_sum((1 - mask) * tf.nn.sigmoid(grid_pred_scores)**2)
  total_loss = object_loss + coord_loss + no_object_loss
  return total_loss, object_loss, coord_loss, no_object_loss
