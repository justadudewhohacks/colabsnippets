import math

import numpy as np

from .utils import box_xywh_to_corner, iou_of, box_corner_to_xywh


class GroundTruthLabelCreatorV1:
  def __init__(self, image_size, anchors_by_stage, stage_num_cells):
    self.image_size = image_size
    self.anchors_by_stage = anchors_by_stage
    self.stage_num_cells = stage_num_cells
    self.pos_iou_threshold = 0.5
    self.neg_iou_threshold = 0.3

  def initialize_anchors(self):
    self.stage_anchor_corner_coords = []
    self.stage_anchor_descriptors = []
    for stage_idx, stage_anchors in enumerate(self.anchors_by_stage):
      num_cells = self.stage_num_cells[stage_idx]
      cell_size = self.image_size / num_cells
      for anchor_idx, anchor in enumerate(stage_anchors):
        aw, ah = anchor
        for col in range(0, num_cells):
          for row in range(0, num_cells):
            cell_x, cell_y = col * cell_size, row * cell_size
            ax_ct = cell_x + (cell_size / 2)
            ay_ct = cell_y + (cell_size / 2)
            ax = ax_ct - (aw / 2)
            ay = ay_ct - (ah / 2)
            anchor_corner_coords = box_xywh_to_corner([ax, ay, aw, ah])
            # print([ax, ay, aw, ah], anchor_corner_coords)
            self.stage_anchor_corner_coords.append(anchor_corner_coords)
            self.stage_anchor_descriptors.append([stage_idx, anchor_idx, col, row])
    self.stage_anchor_corner_coords = np.array(self.stage_anchor_corner_coords)

  def create_gt_masks(self, batch_gt_boxes):
    batch_size = len(batch_gt_boxes)

    pos_anchors_masks_by_stage = []
    neg_anchors_masks_by_stage = []
    offsets_by_stage = []
    scales_by_stage = []
    for stage_idx, num_cells in enumerate(self.stage_num_cells):
      pos_anchors_masks_by_stage.append([])
      neg_anchors_masks_by_stage.append([])
      offsets_by_stage.append([])
      scales_by_stage.append([])
      for anchor_idx in range(0, len(self.anchors_by_stage[stage_idx])):
        pos_anchors_masks_by_stage[stage_idx].append(
          np.zeros([batch_size, num_cells, num_cells, 1]))
        neg_anchors_masks_by_stage[stage_idx].append(
          np.ones([batch_size, num_cells, num_cells, 1]))
        offsets_by_stage[stage_idx].append(
          np.zeros([batch_size, num_cells, num_cells, 2]))
        scales_by_stage[stage_idx].append(
          np.zeros([batch_size, num_cells, num_cells, 2]))

    gt_boxes_with_no_matching_anchors = []
    gt_boxes_with_multiple_matching_anchors = []
    for batch_idx in range(0, batch_size):
      gt_boxes = batch_gt_boxes[batch_idx]
      if len(gt_boxes) == 0:
        continue

      box_to_abs = lambda box: [v * self.image_size for v in box]
      abs_gt_boxes = [box_to_abs(box) for box in gt_boxes]

      abs_gt_boxes_corner = [box_xywh_to_corner(box) for box in abs_gt_boxes]
      # print('num_gt', len(gt_boxes))
      ious = iou_of(np.array([[b] for b in abs_gt_boxes_corner]), self.stage_anchor_corner_coords)

      # print('ious', ious.shape)
      for gt_idx, abs_gt_box in enumerate(abs_gt_boxes):
        positive_anchor_indices = np.where(ious[gt_idx] > self.pos_iou_threshold)[0]
        soft_positive_anchor_indices = np.where(ious[gt_idx] > self.neg_iou_threshold)[0]

        if len(positive_anchor_indices) == 0:
          gt_boxes_with_no_matching_anchors.append(gt_boxes[gt_idx])

        if len(positive_anchor_indices) > 1:
          gt_boxes_with_multiple_matching_anchors.append(gt_boxes[gt_idx])

        # print('self.stage_anchor_corner_coords', self.stage_anchor_corner_coords.shape)
        # print(self.stage_anchor_corner_coords)
        # print('positive_anchor_indices', positive_anchor_indices)
        # print('abs_gt_box', abs_gt_box)
        for idx in positive_anchor_indices:
          stage_idx, anchor_idx, col, row = self.stage_anchor_descriptors[idx]
          ax, ay, aw, ah = box_corner_to_xywh(self.stage_anchor_corner_coords[idx])
          x, y, w, h = abs_gt_box

          cell_size = self.image_size / self.stage_num_cells[stage_idx]
          ax_ct, ay_ct = ax + (aw / 2), ay + (ah / 2)
          x_ct, y_ct = x + (w / 2), y + (h / 2)
          dx = (ax_ct - x_ct) / cell_size
          dy = (ay_ct - y_ct) / cell_size
          dw = math.log(w / aw)
          dh = math.log(h / ah)
          # print('anchor', idx, [ax, ay, aw, ah], [stage_idx, anchor_idx, col, row])
          # print(abs_gt_box, [ax, ay, aw, ah])
          # print('offset deltas', (ax_ct, ay_ct), (x_ct, y_ct), [dx, dy])
          # print('scale deltas', (aw, ah), (w, h), [dw, dh])

          pos_anchors_masks_by_stage[stage_idx][anchor_idx][batch_idx, col, row, :] = 1
          offsets_by_stage[stage_idx][anchor_idx][batch_idx, col, row, :] = [dx, dy]
          scales_by_stage[stage_idx][anchor_idx][batch_idx, col, row, :] = [dw, dh]

        for idx in soft_positive_anchor_indices:
          stage_idx, anchor_idx, col, row = self.stage_anchor_descriptors[idx]
          neg_anchors_masks_by_stage[stage_idx][anchor_idx][batch_idx, col, row, :] = 0

    masks = {
      "pos_anchors_masks_by_stage": pos_anchors_masks_by_stage,
      "neg_anchors_masks_by_stage": neg_anchors_masks_by_stage,
      "offsets_by_stage": offsets_by_stage,
      "scales_by_stage": scales_by_stage,
      "gt_boxes_with_no_matching_anchors": gt_boxes_with_no_matching_anchors,
      "gt_boxes_with_multiple_matching_anchors": gt_boxes_with_multiple_matching_anchors
    }
    return masks
