import math
import tensorflow as tf
import numpy as np

from ... import NeuralNetwork
from ..calculate_iou import calculate_iou
from ..focal_loss import focal_loss
from .generate_anchors import generate_anchors

class FPNBase(NeuralNetwork):
  def __init__(self, name = 'fpn_base', anchors = generate_anchors(num_anchors_per_stage = 3), stage_idx_offset = 0):
    self.anchors = anchors
    self.stage_idx_offset = stage_idx_offset
    super().__init__(self.initialize_weights, name = name)

  def initialize_weights(self, weight_processor):
    pass

  def forward(self, x, batch_size, image_size, out_num_cells = None):
    pass

  def get_num_anchors_per_stage(self):
    return len(self.anchors[0])

  def get_num_stages(self):
    return len(self.anchors)

  def get_num_cells_for_stage(self, image_size, stage_idx):
    return math.ceil(image_size / (4 * 2**(stage_idx + self.stage_idx_offset)))

  def get_anchors_from_predicted_scores(self, scores_by_stage, score_thresh, image_size):
    batch_anchor_boxes = []
    for s in range(0, self.get_num_stages()):
      batch_scores = scores_by_stage[s]
      num_batches, cols, rows = batch_scores.shape[0:3]
      stage_anchors = self.anchors[s]

      if s == 0:
        for b in range(0, num_batches):
          batch_anchor_boxes.append([])


      for b in range(0, num_batches):
        indices = np.where(batch_scores[b] > score_thresh)

        num_preds =  indices[0].shape[0]
        for i in range(0, num_preds):
          col = indices[0][i]
          row = indices[1][i]
          anchor_idx = indices[2][i]
          w, h = stage_anchors[anchor_idx]
          w = w / image_size
          h = h / image_size
          x = col / cols - (w / 2)
          y = row / rows - (h / 2)
          batch_anchor_boxes[b].append([x, y, w, h])
    return batch_anchor_boxes

  def extract_boxes(self, offsets_by_stage, scales_by_stage, scores_by_stage, score_thresh, image_size, relative_coords = False, with_scores = False):
    batch_boxes = []
    for s in range(0, self.get_num_stages()):
      batch_scores = scores_by_stage[s]
      batch_offsets = offsets_by_stage[s]
      batch_scales = scales_by_stage[s]
      num_batches, cols, rows = batch_scores.shape[0:3]
      stage_anchors = self.anchors[s]
      stage_cell_size = image_size / self.get_num_cells_for_stage(image_size, s)

      if s == 0:
        for b in range(0, num_batches):
          batch_boxes.append([])


      for b in range(0, num_batches):
        indices = np.where(batch_scores[b] > score_thresh)
        # TODO ?
        offsets = batch_offsets[b][indices[0 : len(indices) - 1]]
        scales = batch_scales[b][indices[0 : len(indices) - 1]]
        scores = batch_scores[b][indices]

        num_preds =  indices[0].shape[0]
        for i in range(0, num_preds):
          col = indices[0][i]
          row = indices[1][i]
          anchor_idx = indices[2][i]
          aw, ah = stage_anchors[anchor_idx]

          ct_x = (offsets[i][0] + col) * stage_cell_size
          ct_y = (offsets[i][1] + row) * stage_cell_size
          w = scales[i][0] * aw
          h = scales[i][1] * ah
          x = ct_x - w/2
          y = ct_y - h/2

          box = np.array([x, y, w, h])
          if relative_coords:
            box /= image_size

          if with_scores:
            box = box.tolist() + scores[i]

          batch_boxes[b].append(box)
    return batch_boxes

  def get_positive_anchors(self, box, image_size, iou_threshold = 0.5):

    # TODO: what if box centers are out of grid?
    in_grid_range = lambda val, num_cells: min(num_cells - 1, max(0, val))

    x, y, w, h = box
    box_w, box_h = w * image_size, h * image_size

    positive_anchors = []
    for stage_idx, stage_anchors in enumerate(self.anchors):
      for anchor_idx, anchor in enumerate(stage_anchors):
        aw, ah = anchor
        iou = calculate_iou((0, 0, aw, ah), (0, 0, box_w, box_h))
        if iou > iou_threshold:
          ct_x = x + (w / 2)
          ct_y = y + (h / 2)
          stage_num_cells = self.get_num_cells_for_stage(image_size, stage_idx)
          col = in_grid_range(math.floor(ct_x * stage_num_cells), stage_num_cells)
          row = in_grid_range(math.floor(ct_y * stage_num_cells), stage_num_cells)
          positive_anchors.append([stage_idx, col, row, anchor_idx])

    return positive_anchors

  def to_gt_coords(self, gt_box, positive_anchor, image_size):
    stage_idx, col, row, anchor_idx = positive_anchor
    stage_num_cells = self.get_num_cells_for_stage(image_size, stage_idx)
    aw, ah = self.anchors[stage_idx][anchor_idx]

    x, y, w, h = gt_box
    ct_x = x + (w / 2)
    ct_y = y + (h / 2)
    gt_x = (ct_x * stage_num_cells) - col
    gt_y = (ct_y * stage_num_cells) - row
    gt_w = (w * image_size) / aw
    gt_h = (h * image_size) / ah

    #gt_x, gt_y = inverse_sigmoid(min(max(gt_x, 0.001), 0.999)), inverse_sigmoid(min(max(gt_y, 0.001), 0.999))
    #gt_w, gt_h = math.log(gt_w), math.log(gt_h)

    return [gt_x, gt_y, gt_w, gt_h]

  def create_gt_masks(self, batch_gt_boxes, image_size, is_return_grid_cell_masks = False):
    batch_size = len(batch_gt_boxes)

    anchor_masks_by_stage = []
    grid_cell_masks_by_stage = []
    offsets_by_stage = []
    scales_by_stage = []
    for stage_idx in range(0, self.get_num_stages()):
      stage_num_cells = self.get_num_cells_for_stage(image_size, stage_idx)
      anchor_masks_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 1]))
      grid_cell_masks_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, 1]))
      offsets_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 2]))
      scales_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 2]))

    for batch_idx in range(0, batch_size):
      for gt_box in batch_gt_boxes[batch_idx]:
        positive_anchors = self.get_positive_anchors(gt_box, image_size)

        #if len(positive_anchors) == 0:
          #print('warning, no positive_anchors for box: ' + str(gt_box))

        #if len(positive_anchors) > 1:
          #print('warning, ' + str(len(positive_anchors)) + ' positive_anchors for box: ' + str(gt_box))

        for positive_anchor in positive_anchors:
          stage_idx, col, row, anchor_idx = positive_anchor
          anchor_masks_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = 1
          grid_cell_masks_by_stage[stage_idx][batch_idx, col, row, :] = 1
          gt_x, gt_y, gt_w, gt_h = self.to_gt_coords(gt_box, positive_anchor, image_size)
          offsets_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = [gt_x, gt_y]
          scales_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = [gt_w, gt_h]

    masks = [anchor_masks_by_stage, offsets_by_stage, scales_by_stage]
    if is_return_grid_cell_masks:
      masks += grid_cell_masks_by_stage
    return masks

  def create_gt_masks_v2(self, batch_gt_boxes, image_size):
    batch_size = len(batch_gt_boxes)

    anchor_masks_by_stage = []
    grid_cell_masks_by_stage = []
    offsets_by_stage = []
    scales_by_stage = []
    for stage_idx in range(0, self.get_num_stages()):
      stage_num_cells = self.get_num_cells_for_stage(image_size, stage_idx)
      anchor_masks_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 1]))
      grid_cell_masks_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 1]))
      offsets_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 2]))
      scales_by_stage.append(np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 2]))

    for batch_idx in range(0, batch_size):
      for gt_box in batch_gt_boxes[batch_idx]:
        positive_anchors = self.get_positive_anchors(gt_box, image_size)

        #if len(positive_anchors) == 0:
          #print('warning, no positive_anchors for box: ' + str(gt_box))

        #if len(positive_anchors) > 1:
          #print('warning, ' + str(len(positive_anchors)) + ' positive_anchors for box: ' + str(gt_box))

        for positive_anchor in positive_anchors:
          stage_idx, col, row, anchor_idx = positive_anchor
          anchor_masks_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = 1
          grid_cell_masks_by_stage[stage_idx][batch_idx, col, row, :, :] = 1
          gt_x, gt_y, gt_w, gt_h = self.to_gt_coords(gt_box, positive_anchor, image_size)
          offsets_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = [gt_x, gt_y]
          scales_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = [gt_w, gt_h]

    masks = {
      "anchor_masks_by_stage": anchor_masks_by_stage,
      "grid_cell_masks_by_stage": grid_cell_masks_by_stage,
      "offsets_by_stage": offsets_by_stage,
      "scales_by_stage": scales_by_stage,
    }
    return masks

  def coords_and_scores(self, x, stage_num_cells, batch_size):
    out = tf.reshape(x, [batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 5])
    offsets = tf.slice(out, [0, 0, 0, 0, 0], [batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 2])
    scales = tf.slice(out, [0, 0, 0, 0, 2], [batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 2])
    scores = tf.slice(out, [0, 0, 0, 0, 4], [batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_per_stage(), 1])
    return tf.sigmoid(offsets), scales, scores

  def forward_factory(self, sess, batch_size, image_size, out_num_cells = 20, with_train_ops = False,
                      object_scale = 1.0, coord_scale = 1.0, no_object_scale = 0.25, learning_rate = None):
    X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])

    stages_ops = self.forward(X, batch_size, image_size, out_num_cells = out_num_cells)
    offsets_ops_by_stage = [ops[0] for ops in stages_ops]
    scales_ops_by_stage = [ops[1] for ops in stages_ops]
    scores_ops_by_stage = [tf.nn.sigmoid(ops[2]) for ops in stages_ops]

    def forward(batch_x, score_thresh = 0.5):
      feed_dict = { X: batch_x }
      batch_offsets_by_stage, batch_scales_by_stage , batch_scores_by_stage, = sess.run([offsets_ops_by_stage, scales_ops_by_stage, scores_ops_by_stage], feed_dict = feed_dict)
      return self.extract_boxes(batch_offsets_by_stage, batch_scales_by_stage, batch_scores_by_stage, score_thresh, image_size, relative_coords = True)

    if with_train_ops:
      num_stages = self.get_num_stages()
      num_anchors_per_stage = self.get_num_anchors_per_stage()
      num_cells_by_stage = [self.get_num_cells_for_stage(image_size, s) for s in range(0, num_stages)]
      MASKS_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, num_anchors_per_stage, 1]) for nc in num_cells_by_stage]
      OFFSETS_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, num_anchors_per_stage, 2]) for nc in num_cells_by_stage]
      SCALES_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, num_anchors_per_stage, 2]) for nc in num_cells_by_stage]

      object_loss_ops_by_stage = [tf.reduce_sum(MASKS_BY_STAGE[s] * focal_loss(scores_ops_by_stage[s], True)) for s in range(0, num_stages)]
      no_object_loss_ops_by_stage = [tf.reduce_sum((1 - MASKS_BY_STAGE[s]) * focal_loss(scores_ops_by_stage[s], False)) for s in range(0, num_stages)]

      offset_loss_ops_by_stage = [tf.reduce_sum((offsets_ops_by_stage[s] - OFFSETS_BY_STAGE[s])**2 * MASKS_BY_STAGE[s]) for s in range(0, num_stages)]
      scales_loss_ops_by_stage = [tf.reduce_sum((scales_ops_by_stage[s] - SCALES_BY_STAGE[s])**2 * MASKS_BY_STAGE[s]) for s in range(0, num_stages)]

      object_loss_op = tf.add_n(object_loss_ops_by_stage)
      no_object_loss_op = tf.add_n(no_object_loss_ops_by_stage)
      offsets_loss_op = tf.add_n(offset_loss_ops_by_stage)
      scales_loss_op = tf.add_n(scales_loss_ops_by_stage)

      object_loss_op = object_scale * object_loss_op
      no_object_loss_op = no_object_scale * no_object_loss_op
      offsets_loss_op = coord_scale * offsets_loss_op
      scales_loss_op = coord_scale * scales_loss_op

      loss_op = (object_loss_op + no_object_loss_op + offsets_loss_op + scales_loss_op) / batch_size
      train_op = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'Adam_' + str(image_size)).minimize(loss_op)

      def forward_train(batch_x, batch_gt_boxes, score_thresh = 0.5):
        gt_masks_by_stage, gt_offsets_by_stage, gt_scales_by_stage = self.create_gt_masks(batch_gt_boxes, image_size)

        feed_dict = { X: batch_x }
        for s in range(0, num_stages):
          feed_dict[MASKS_BY_STAGE[s]] = gt_masks_by_stage[s]
          feed_dict[OFFSETS_BY_STAGE[s]] = gt_offsets_by_stage[s]
          feed_dict[SCALES_BY_STAGE[s]] = gt_scales_by_stage[s]

        loss, _, object_loss, no_object_loss, offsets_loss, scales_loss, batch_scores_by_stage, batch_offsets_by_stage, batch_scales_by_stage = sess.run([
            loss_op, train_op, object_loss_op, no_object_loss_op, offsets_loss_op, scales_loss_op,
            scores_ops_by_stage, offsets_ops_by_stage, scales_ops_by_stage
        ], feed_dict = feed_dict)

        batch_pred_boxes = self.extract_boxes(batch_offsets_by_stage, batch_scales_by_stage, batch_scores_by_stage, score_thresh, image_size, relative_coords = True)

        ret = {
            "loss": loss,
            "object_loss": object_loss,
            "no_object_loss": no_object_loss,
            "offsets_loss": offsets_loss,
            "scales_loss": scales_loss,
            "gt_masks_by_stage": gt_masks_by_stage,
            "batch_scores_by_stage": batch_scores_by_stage,
            "batch_pred_boxes": batch_pred_boxes
        }
        return ret
      return forward_train
    return forward
