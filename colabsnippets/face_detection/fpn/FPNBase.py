import math

import numpy as np
import tensorflow as tf

from .generate_anchors import generate_anchors
from ..calculate_iou import calculate_iou
from ..focal_loss import focal_loss
from ... import NeuralNetwork


def batch_boxes_by_stage_to_boxes_by_batch(batch_boxes_by_stage):
  batch_size = len(batch_boxes_by_stage[0])
  out_batch_boxes = [[] for b in range(0, batch_size)]
  for batch_boxes in batch_boxes_by_stage:
    for batch_idx, boxes in enumerate(batch_boxes):
      for box in boxes:
        out_batch_boxes[batch_idx].append(box)
  return out_batch_boxes


class FPNBase(NeuralNetwork):
  def __init__(self, name='fpn_base', anchors=generate_anchors(num_anchors_per_stage=3), stage_idx_offset=0):
    self.anchors = anchors
    self.stage_idx_offset = stage_idx_offset
    super().__init__(self.initialize_weights, name=name)

  def initialize_weights(self, weight_processor):
    raise Exception("FPNBase - initialize_weights not implemented")

  def forward(self, X, batch_size, image_size, out_num_cells=None):
    raise Exception("FPNBase - forward not implemented")

  def forward_and_get_ops_by_stage(self, X, batch_size, image_size, out_num_cells=None):
    stages_ops = self.forward(X, batch_size, image_size, out_num_cells=out_num_cells)
    offsets_ops_by_stage = [ops[0] for ops in stages_ops]
    scales_ops_by_stage = [ops[1] for ops in stages_ops]
    scores_ops_by_stage = [tf.nn.sigmoid(ops[2]) for ops in stages_ops]
    return {
      "offsets_ops_by_stage": offsets_ops_by_stage,
      "scales_ops_by_stage": scales_ops_by_stage,
      "scores_ops_by_stage": scores_ops_by_stage
    }

  def get_num_anchors_for_stage(self, stage_idx):
    return len(self.anchors[stage_idx])

  def get_num_stages(self):
    return len(self.anchors)

  def get_num_cells_for_stage(self, image_size, stage_idx):
    return math.ceil(image_size / (4 * 2 ** (stage_idx + self.stage_idx_offset)))

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

        num_preds = indices[0].shape[0]
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

  def extract_boxes(self, offsets_by_stage, scales_by_stage, scores_by_stage, score_thresh, image_size,
                    relative_coords=False, with_scores=False):
    batch_boxes_by_stage = self.extract_boxes_by_stage(offsets_by_stage, scales_by_stage, scores_by_stage, score_thresh,
                                                       image_size, relative_coords=relative_coords,
                                                       with_scores=with_scores)
    return batch_boxes_by_stage_to_boxes_by_batch(batch_boxes_by_stage)

  def extract_boxes_by_stage(self, offsets_by_stage, scales_by_stage, scores_by_stage, score_thresh, image_size,
                             relative_coords=False, with_scores=False):
    batch_boxes_by_stage = [[] for s in range(0, self.get_num_stages())]
    for s in range(0, self.get_num_stages()):
      batch_scores = scores_by_stage[s]
      batch_offsets = offsets_by_stage[s]
      batch_scales = scales_by_stage[s]
      num_batches, cols, rows = batch_scores.shape[0:3]
      stage_anchors = self.anchors[s]
      stage_cell_size = image_size / self.get_num_cells_for_stage(image_size, s)

      for b in range(0, num_batches):
        batch_boxes_by_stage[s].append([])
        indices = np.where(batch_scores[b] > score_thresh)
        offsets = batch_offsets[b][indices[0: len(indices) - 1]]
        scales = batch_scales[b][indices[0: len(indices) - 1]]
        scores = batch_scores[b][indices]

        num_preds = indices[0].shape[0]
        for i in range(0, num_preds):
          col = indices[0][i]
          row = indices[1][i]
          anchor_idx = indices[2][i]
          aw, ah = stage_anchors[anchor_idx]

          ct_x = (offsets[i][0] + col) * stage_cell_size
          ct_y = (offsets[i][1] + row) * stage_cell_size
          w = scales[i][0] * aw
          h = scales[i][1] * ah
          x = ct_x - w / 2
          y = ct_y - h / 2

          box = np.array([x, y, w, h])
          if relative_coords:
            box /= image_size

          if with_scores:
            box = [v for v in box]
            box.append(scores[i])

          batch_boxes_by_stage[s][b].append(box)
    return batch_boxes_by_stage

  def get_positive_anchors(self, box, image_size, iou_threshold=0.5):

    # TODO: what if box centers are out of grid?
    in_grid_range = lambda val, num_cells: min(num_cells - 1, max(0, val))

    x, y, w, h = box
    box_w, box_h = w * image_size, h * image_size

    positive_anchors = []
    for stage_idx, stage_anchors in enumerate(self.anchors):
      for anchor_idx, anchor in enumerate(stage_anchors):
        aw, ah = anchor
        iou = calculate_iou((0, 0, aw, ah), (0, 0, box_w, box_h))
        if iou >= iou_threshold:
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

    # gt_x, gt_y = inverse_sigmoid(min(max(gt_x, 0.001), 0.999)), inverse_sigmoid(min(max(gt_y, 0.001), 0.999))
    # gt_w, gt_h = math.log(gt_w), math.log(gt_h)

    return [gt_x, gt_y, gt_w, gt_h]

  def create_gt_masks(self, batch_gt_boxes, image_size, pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    batch_size = len(batch_gt_boxes)

    pos_anchors_masks_by_stage = []
    neg_anchors_masks_by_stage = []
    offsets_by_stage = []
    scales_by_stage = []
    for stage_idx in range(0, self.get_num_stages()):
      stage_num_cells = self.get_num_cells_for_stage(image_size, stage_idx)
      pos_anchors_masks_by_stage.append(
        np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_for_stage(stage_idx), 1]))
      neg_anchors_masks_by_stage.append(
        np.ones([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_for_stage(stage_idx), 1]))
      offsets_by_stage.append(
        np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_for_stage(stage_idx), 2]))
      scales_by_stage.append(
        np.zeros([batch_size, stage_num_cells, stage_num_cells, self.get_num_anchors_for_stage(stage_idx), 2]))

    gt_boxes_with_no_matching_anchors = []
    gt_boxes_with_multiple_matching_anchors = []
    for batch_idx in range(0, batch_size):
      for gt_box in batch_gt_boxes[batch_idx]:
        positive_anchors = self.get_positive_anchors(gt_box, image_size, iou_threshold=pos_iou_threshold)
        soft_positive_anchors = self.get_positive_anchors(gt_box, image_size, iou_threshold=neg_iou_threshold)

        if len(positive_anchors) == 0:
          gt_boxes_with_no_matching_anchors.append(gt_box)

        if len(positive_anchors) > 1:
          gt_boxes_with_multiple_matching_anchors.append(gt_box)

        for positive_anchor in positive_anchors:
          stage_idx, col, row, anchor_idx = positive_anchor
          pos_anchors_masks_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = 1
          gt_x, gt_y, gt_w, gt_h = self.to_gt_coords(gt_box, positive_anchor, image_size)
          offsets_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = [gt_x, gt_y]
          scales_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = [gt_w, gt_h]

        for soft_positive_anchor in soft_positive_anchors:
          stage_idx, col, row, anchor_idx = soft_positive_anchor
          neg_anchors_masks_by_stage[stage_idx][batch_idx, col, row, anchor_idx, :] = 0

    masks = {
      "pos_anchors_masks_by_stage": pos_anchors_masks_by_stage,
      "neg_anchors_masks_by_stage": neg_anchors_masks_by_stage,
      "offsets_by_stage": offsets_by_stage,
      "scales_by_stage": scales_by_stage,
      "gt_boxes_with_no_matching_anchors": gt_boxes_with_no_matching_anchors,
      "gt_boxes_with_multiple_matching_anchors": gt_boxes_with_multiple_matching_anchors
    }
    return masks

  def coords_and_scores(self, x, image_size, batch_size, stage_idx):
    stage_num_cells = self.get_num_cells_for_stage(image_size, stage_idx)
    stage_num_anchors = self.get_num_anchors_for_stage(stage_idx)
    out = tf.reshape(x, [batch_size, stage_num_cells, stage_num_cells, stage_num_anchors, 5])
    offsets = tf.slice(out, [0, 0, 0, 0, 0],
                       [batch_size, stage_num_cells, stage_num_cells, stage_num_anchors, 2])
    scales = tf.slice(out, [0, 0, 0, 0, 2],
                      [batch_size, stage_num_cells, stage_num_cells, stage_num_anchors, 2])
    scores = tf.slice(out, [0, 0, 0, 0, 4],
                      [batch_size, stage_num_cells, stage_num_cells, stage_num_anchors, 1])
    return tf.sigmoid(offsets), scales, scores

  def forward_factory(self, sess, batch_size, image_size, out_num_cells=20):
    X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    ops_by_stage = self.forward_and_get_ops_by_stage(X, batch_size, image_size, out_num_cells=out_num_cells)

    def forward(batch_x, score_thresh=0.5, with_scores=False):
      batch_offsets_by_stage, batch_scales_by_stage, batch_scores_by_stage, = sess.run(
        [ops_by_stage["offsets_ops_by_stage"], ops_by_stage["scales_ops_by_stage"],
         ops_by_stage["scores_ops_by_stage"]], feed_dict={X: batch_x})
      return self.extract_boxes(batch_offsets_by_stage, batch_scales_by_stage, batch_scores_by_stage, score_thresh,
                                image_size, relative_coords=True, with_scores=with_scores)

    return forward

  def forward_train_factory(self, sess, batch_size, image_size, out_num_cells=20,
                            object_scale=1.0, coord_scale=1.0, no_object_scale=0.25, offsets_scale=None,
                            scales_scale=None, apply_scale_loss=lambda x: x, compile_optimizer_op=None,
                            stage_object_loss_scales=None):
    offsets_scale = coord_scale if offsets_scale is None else offsets_scale
    scales_scale = coord_scale if scales_scale is None else scales_scale
    X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    ops_by_stage = self.forward_and_get_ops_by_stage(X, batch_size, image_size, out_num_cells=out_num_cells)

    num_stages = self.get_num_stages()
    num_anchors_and_cells_by_stage = [
      [self.get_num_anchors_for_stage(stage_idx), self.get_num_cells_for_stage(image_size, stage_idx)] for stage_idx
      in range(0, num_stages)]
    POS_ANCHORS_MASKS_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, a, 1]) for a, nc in
                                  num_anchors_and_cells_by_stage]
    NEG_ANCHORS_MASKS_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, a, 1]) for a, nc in
                                  num_anchors_and_cells_by_stage]
    OFFSETS_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, a, 2]) for a, nc in
                        num_anchors_and_cells_by_stage]
    SCALES_BY_STAGE = [tf.placeholder(tf.float32, [batch_size, nc, nc, a, 2]) for a, nc in
                       num_anchors_and_cells_by_stage]

    object_loss_ops_by_stage = [
      tf.reduce_sum(POS_ANCHORS_MASKS_BY_STAGE[s] * focal_loss(ops_by_stage["scores_ops_by_stage"][s], True)) for s in
      range(0, num_stages)]
    no_object_loss_ops_by_stage = [
      tf.reduce_sum(NEG_ANCHORS_MASKS_BY_STAGE[s] * focal_loss(ops_by_stage["scores_ops_by_stage"][s], False))
      for s in range(0, num_stages)]

    offset_loss_ops_by_stage = [
      tf.reduce_sum(
        (ops_by_stage["offsets_ops_by_stage"][s] - OFFSETS_BY_STAGE[s]) ** 2 * POS_ANCHORS_MASKS_BY_STAGE[s]) for s in
      range(0, num_stages)]
    scales_loss_ops_by_stage = [
      tf.reduce_sum(
        apply_scale_loss((ops_by_stage["scales_ops_by_stage"][s] - SCALES_BY_STAGE[s])) * POS_ANCHORS_MASKS_BY_STAGE[s])
      for s in
      range(0, num_stages)]

    # TODO stage scale factors
    stage_object_loss_scales = [1.0 for s in
                                range(0, num_stages)] if stage_object_loss_scales is None else stage_object_loss_scales
    if len(stage_object_loss_scales) != num_stages:
      raise Exception(
        "len(stage_object_loss_scales)= {}, but num_stages is {}".format(len(stage_object_loss_scales), num_stages))
    compute_weighted_losses = lambda loss_ops_by_stage, scale, stage_loss_scales: [
      (scale * l * stage_loss_scales[stage_idx]) / batch_size
      for
      stage_idx, l in enumerate(loss_ops_by_stage)]
    weighted_object_loss_ops_by_stage = compute_weighted_losses(object_loss_ops_by_stage, object_scale,
                                                                stage_object_loss_scales)
    weighted_no_object_loss_ops_by_stage = compute_weighted_losses(no_object_loss_ops_by_stage, no_object_scale,
                                                                   [1.0 for s in range(0, num_stages)])
    weighted_offset_loss_ops_by_stage = compute_weighted_losses(offset_loss_ops_by_stage, offsets_scale,
                                                                [1.0 for s in range(0, num_stages)])
    weighted_scales_loss_ops_by_stage = compute_weighted_losses(scales_loss_ops_by_stage, scales_scale,
                                                                [1.0 for s in range(0, num_stages)])

    object_loss_op = tf.add_n(weighted_object_loss_ops_by_stage)
    no_object_loss_op = tf.add_n(weighted_no_object_loss_ops_by_stage)
    offsets_loss_op = tf.add_n(weighted_offset_loss_ops_by_stage)
    scales_loss_op = tf.add_n(weighted_scales_loss_ops_by_stage)
    loss_op = object_loss_op + no_object_loss_op + offsets_loss_op + scales_loss_op

    if compile_optimizer_op is None:
      raise Exception("compile_optimizer_op is not optional")
    train_op = compile_optimizer_op(loss_op, image_size)

    # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_' + str(image_size)).minimize(loss_op)

    def forward_train(batch_x, batch_gt_boxes, score_thresh=0.5, with_scores=False):
      masks = self.create_gt_masks(batch_gt_boxes, image_size)

      feed_dict = {X: batch_x}
      for s in range(0, num_stages):
        feed_dict[POS_ANCHORS_MASKS_BY_STAGE[s]] = masks["pos_anchors_masks_by_stage"][s]
        feed_dict[NEG_ANCHORS_MASKS_BY_STAGE[s]] = masks["neg_anchors_masks_by_stage"][s]
        feed_dict[OFFSETS_BY_STAGE[s]] = masks["offsets_by_stage"][s]
        feed_dict[SCALES_BY_STAGE[s]] = masks["scales_by_stage"][s]

      sess_ret = sess.run([
        loss_op, train_op,
        weighted_object_loss_ops_by_stage, weighted_no_object_loss_ops_by_stage, weighted_offset_loss_ops_by_stage,
        weighted_scales_loss_ops_by_stage,
        ops_by_stage["scores_ops_by_stage"], ops_by_stage["offsets_ops_by_stage"], ops_by_stage["scales_ops_by_stage"]
      ], feed_dict=feed_dict)

      loss, _, weighted_object_losses_by_stage, weighted_no_object_losses_by_stage, weighted_offset_losses_by_stage, weighted_scales_losses_by_stage, batch_scores_by_stage, batch_offsets_by_stage, batch_scales_by_stage, = sess_ret

      batch_pred_boxes_by_stage = self.extract_boxes_by_stage(batch_offsets_by_stage, batch_scales_by_stage,
                                                              batch_scores_by_stage, score_thresh, image_size,
                                                              relative_coords=True, with_scores=with_scores)

      ret = {
        "loss": loss,
        "weighted_object_losses_by_stage": weighted_object_losses_by_stage,
        "weighted_no_object_losses_by_stage": weighted_no_object_losses_by_stage,
        "weighted_offset_losses_by_stage": weighted_offset_losses_by_stage,
        "weighted_scales_losses_by_stage": weighted_scales_losses_by_stage,
        "pos_anchors_masks_by_stage": masks["pos_anchors_masks_by_stage"],
        "neg_anchors_masks_by_stage": masks["neg_anchors_masks_by_stage"],
        "batch_scores_by_stage": batch_scores_by_stage,
        "batch_pred_boxes_by_stage": batch_pred_boxes_by_stage,
        "gt_boxes_with_no_matching_anchors": masks["gt_boxes_with_no_matching_anchors"],
        "gt_boxes_with_multiple_matching_anchors": masks["gt_boxes_with_multiple_matching_anchors"]
      }
      return ret

    return forward_train
