import tensorflow as tf
import numpy as np

from ... import NeuralNetwork
from ..calculate_iou import calculate_iou
from .commons import get_box_center, get_cell_position_of_box, get_gt_coords

class ObjectDetectorBase(NeuralNetwork):
  def __init__(self, name = 'object_detector_base', num_cells = 20):
    self.num_cells = num_cells
    super().__init__(self.initialize_weights, name = name)

  def initialize_weights(self, weight_processor):
    pass

  def create_gt_masks(self, batch_gt_boxes, image_size):
    batch_size = len(batch_gt_boxes)

    batch_grid_cell_mask = np.zeros([batch_size, self.num_cells, self.num_cells, 1])
    batch_gt_coords = np.zeros([batch_size, self.num_cells, self.num_cells, 2])
    batch_gt_scales = np.zeros([batch_size, self.num_cells, self.num_cells, 2])

    for batch_idx in range(0, batch_size):
      for gt_box in batch_gt_boxes[batch_idx]:
        col, row = get_cell_position_of_box(gt_box, self.num_cells)
        gt_x, gt_y = get_gt_coords(gt_box, self.num_cells)
        _, __, w, h = gt_box
        batch_grid_cell_mask[batch_idx, col, row, :] = 1
        batch_gt_coords[batch_idx, col, row, :] = [gt_x, gt_y]
        batch_gt_scales[batch_idx, col, row, :] = [w, h]

    return batch_grid_cell_mask, batch_gt_coords, batch_gt_scales

  def get_boxes_and_scores_tensors(self, batch_pred, num_cells, batch_size):
    out = tf.reshape(batch_pred, [batch_size, num_cells, num_cells, 5])
    batch_pred_coords = tf.slice(out, [0, 0, 0, 0], [batch_size, num_cells, num_cells, 2])
    batch_pred_scales = tf.slice(out, [0, 0, 0, 2], [batch_size, num_cells, num_cells, 2])
    batch_pred_scores = tf.slice(out, [0, 0, 0, 4], [batch_size, num_cells, num_cells, 1])
    return tf.sigmoid(batch_pred_coords), tf.sigmoid(batch_pred_scales), tf.sigmoid(batch_pred_scores)

  def extract_boxes(self, batch_offsets, batch_scales, batch_scores, score_thresh, image_size, relative_coords = False, with_scores = False):
    batch_size = batch_scores.shape[0]
    batch_boxes = []

    for b in range(0, batch_size):
      indices = np.where(batch_scores[b] > score_thresh)

      offsets = batch_offsets[b][indices[0 : len(indices) - 1]]
      scales = batch_scales[b][indices[0 : len(indices) - 1]]
      scores = batch_scores[b][indices]

      num_preds =  indices[0].shape[0]
      for i in range(0, num_preds):
        col = indices[0][i]
        row = indices[1][i]
        ct_x = (offsets[i][0] + col) / self.num_cells
        ct_y = (offsets[i][1] + row) / self.num_cells
        w = scales[i][0]
        h = scales[i][1]
        x = ct_x - w / 2
        y = ct_y - h / 2

        box = [x, y, w, h]
        if with_scores:
          box += scores[i]

        batch_boxes[b].append(box)
    return batch_boxes