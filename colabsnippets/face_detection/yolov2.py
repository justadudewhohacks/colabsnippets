import math
import numpy as np
import tensorflow as tf
from .calculate_iou import calculate_iou
from .sigmoid import inverse_sigmoid, sigmoid

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

def create_gt_coords(batch_gt_boxes, num_cells, anchors, is_apply_inverse_sigmoid = False, is_activate_coordinates = True):
  batch_size = len(batch_gt_boxes)
  gt_coords = np.zeros([batch_size, num_cells, num_cells, len(anchors), 4])
  for batch_idx in range(0, batch_size):
    for gt_box in batch_gt_boxes[batch_idx]:
      col, row, anchor_idx = get_box_grid_position(gt_box, num_cells, anchors)

      x, y, w, h = gt_box
      ct_x = x + (w / 2)
      ct_y = y + (h / 2)
      aw, ah = anchors[anchor_idx]
      gt_x = (ct_x * num_cells) - col
      gt_y = (ct_y * num_cells) - row
      gt_w = (w * num_cells) / aw
      gt_h = (h * num_cells) / ah

      if is_activate_coordinates:
        if is_apply_inverse_sigmoid:
          gt_x, gt_y = inverse_sigmoid(max(gt_x, 0.001)), inverse_sigmoid(max(gt_y, 0.001))
        gt_w, gt_h = math.log(gt_w), math.log(gt_h)

      gt_coords[batch_idx, col, row, anchor_idx, :] = [gt_x, gt_y, gt_w, gt_h]

  return gt_coords

def extract_coords_and_scores(pred):
  num_anchors = pred.shape.as_list()[3] / 5
  get_shape = lambda size: np.concatenate((pred.shape.as_list()[0:3], [num_anchors, size]), axis = None)
  grid_preds = tf.reshape(pred, get_shape(5))
  grid_pred_coords = tf.slice(grid_preds, [0, 0, 0, 0, 0], get_shape(4))
  grid_pred_scores = tf.slice(grid_preds, [0, 0, 0, 0, 4], get_shape(1))
  return grid_pred_coords, grid_pred_scores

def reconstruct_box(pred_box, col, row, anchor, num_cells, is_apply_sigmoid = False):
  aw, ah = anchor
  x, y, w, h = pred_box
  w = (math.exp(w) * aw) / num_cells
  h = (math.exp(h) * ah) / num_cells
  if is_apply_sigmoid:
    x, y = sigmoid(x), sigmoid(y)
  x = ((col + x) / num_cells) - (w / 2)
  y = ((row + y) / num_cells) - (h / 2)
  return x, y, w, h

def extract_boxes(grid_pred_coords, grid_pred_scores, anchors, min_score = 0.5, is_apply_sigmoid = False):
  batch_size, num_cells = grid_pred_scores.shape[0:2]

  batch_out_boxes = []
  for batch_idx in range(0, batch_size):
    out_boxes = []
    for col in range(0, num_cells):
      for row in range(0, num_cells):
        for anchor_idx in range(0, len(anchors)):
          score = grid_pred_scores[batch_idx, col, row, anchor_idx]
          if score >= min_score:
            box = grid_pred_coords[batch_idx, col, row, anchor_idx]
            out_boxes.append(reconstruct_box(box, col, row, anchors[anchor_idx], num_cells, is_apply_sigmoid = is_apply_sigmoid))
    batch_out_boxes.append(out_boxes)

  return batch_out_boxes

def compile_loss_op(pred, gt_coords, mask, coord_scale = 1.0, object_scale = 5.0, no_object_scale = 1.0):
  grid_pred_coords, grid_pred_scores = extract_coords_and_scores(pred)
  # TODO: ious
  ious = 1
  object_loss = object_scale * tf.reduce_sum(mask * (ious - tf.nn.sigmoid(grid_pred_scores))**2)
  coord_loss = coord_scale * tf.reduce_sum(mask * (grid_pred_coords - gt_coords)**2)
  no_object_loss = no_object_scale * tf.reduce_sum((1 - mask) * tf.nn.sigmoid(grid_pred_scores)**2)
  total_loss = object_loss + coord_loss + no_object_loss
  return total_loss, object_loss, coord_loss, no_object_loss
