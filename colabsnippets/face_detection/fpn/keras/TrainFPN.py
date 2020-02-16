import math

import numpy as np
from keras.utils import Sequence

from colabsnippets.face_detection.fpn.DataLoader import DataLoader
from colabsnippets.face_detection.fpn.keras.GroundTruthLabelCreator import GroundTruthLabelCreator
from colabsnippets.utils import try_upload_file, load_json


class TrainFPN(Sequence):
  def __init__(self, image_augmentor, augmentation_prob, batch_size, image_size, anchors_by_stage, model, epoch=0,
               is_wider_only=False, is_exclude_wider=False, with_mafa_and_ufdd=False, with_celeba_face_clusters=False,
               model_name='none',
               stats_boxes_category_sizes=[0, 8, 16, 32, 64, 128, 256, 512]):
    self.batch_size = batch_size
    self.image_size = image_size
    self.anchors_by_stage = anchors_by_stage
    self.model = model
    self.epoch = epoch
    self.model_name = model_name
    self.min_box_size_px = 8
    self.is_wider_only = is_wider_only
    self.is_exclude_wider = is_exclude_wider
    self.with_mafa_and_ufdd = with_mafa_and_ufdd
    self.with_celeba_face_clusters = with_celeba_face_clusters
    self.stats_boxes_category_sizes = stats_boxes_category_sizes

    wider_trainData = load_json('./wider_trainData.json')
    if self.is_wider_only:
      train_data = wider_trainData
    else:
      ibug_challenge_data = load_json('./ibug_challenge_data.json')
      face_detection_scrapeddb_data = load_json('./face_detection_scrapeddb_data.json')
      train_data = wider_trainData + ibug_challenge_data + face_detection_scrapeddb_data
      if self.is_exclude_wider:
        train_data = ibug_challenge_data + face_detection_scrapeddb_data
    if self.with_mafa_and_ufdd:
      mafa = load_json('./MAFA_train_data.json')
      ufdd = load_json('./UFDD_val_data.json')
      for d in mafa:
        train_data.append(d)
      for d in ufdd:
        train_data.append(d)
    if self.with_celeba_face_clusters:
      celeba_face_clusters = load_json('./celeba_face_clusters_data.json')
      for d in celeba_face_clusters:
        train_data.append(d)

    self.train_data_loader = DataLoader(train_data, start_epoch=self.epoch, image_augmentor=image_augmentor,
                                        augmentation_prob=augmentation_prob, min_box_size_px=0)
    self.reset_epoch_stats()
    print('train samples:', len(train_data))

  def initialize(self, stage_num_cells, pos_iou_threshold=0.5, neg_iou_threshold=0.3, grid_expand_n_at_borders=0):
    self.ground_truth_label_creator = GroundTruthLabelCreator(self.image_size, self.anchors_by_stage, stage_num_cells,
                                                              pos_iou_threshold, neg_iou_threshold,
                                                              grid_expand_n_at_borders=grid_expand_n_at_borders)

  def print_epoch_stats(self):
    print(self.epoch_stats)
    print('stats_boxes_category_sizes', self.stats_boxes_category_sizes)

  def reset_epoch_stats(self):
    self.epoch_stats = {
      "iteration_count": 0,
      "num_outsourced": 0,
      "num_gt_boxes": np.zeros([len(self.stats_boxes_category_sizes)], int),
      "num_unmatched_gt_boxes": np.zeros([len(self.stats_boxes_category_sizes)], int),
      "matched_by_stage_idx": np.zeros([len(self.anchors_by_stage)], int),
      "anchor_multiple_num_matches_by_stage_idx": np.zeros([len(self.anchors_by_stage)], int)
    }

  def get_box_category_idx(self, abs_box):
    w, h = abs_box[2:]
    for idx, size in enumerate(self.stats_boxes_category_sizes):
      if (w * h) < (size * size):
        idx -= 1
        break
    return idx

  def update_epoch_stats(self, batch_filtered_gt_boxes, masks, num_outsourced):
    num_stages = len(self.anchors_by_stage)
    matched_by_stage_idx = []
    anchor_multiple_num_matches_by_stage_idx = []
    for stage_idx in range(0, num_stages):
      for anchor_idx in range(0, len(self.anchors_by_stage[stage_idx])):
        pos_anchors_mask = masks["pos_anchors_masks_by_stage"][stage_idx][anchor_idx]
        anchor_num_matches = masks["anchor_num_matches_by_stage"][stage_idx][anchor_idx]
        matched_by_stage_idx.append(np.where(pos_anchors_mask > 0)[0].shape[0])
        anchor_multiple_num_matches_by_stage_idx.append(np.where(anchor_num_matches > 1)[0].shape[0])

    self.epoch_stats["iteration_count"] += self.batch_size
    self.epoch_stats["num_outsourced"] += num_outsourced
    self.epoch_stats["matched_by_stage_idx"] += np.array(matched_by_stage_idx)
    self.epoch_stats["anchor_multiple_num_matches_by_stage_idx"] += np.array(anchor_multiple_num_matches_by_stage_idx)

    for batch_idx in range(0, self.batch_size):
      abs_gt_boxes = [np.array(box) * self.image_size for box in batch_filtered_gt_boxes[batch_idx]]
      box_matches = [len(matches) for matches in masks["batch_gt_boxes_matches"][batch_idx]]

      for box_idx, abs_gt_box in enumerate(abs_gt_boxes):
        box_category_idx = self.get_box_category_idx(abs_gt_box)
        self.epoch_stats["num_gt_boxes"][box_category_idx] += 1
        if box_matches[box_idx] < 1:
          self.epoch_stats["num_unmatched_gt_boxes"][box_category_idx] += 1

  def __len__(self):
    return int(math.ceil(self.get_data_loader().get_end_idx() / self.batch_size))

  def __getitem__(self, idx):
    batch_x, batch_gt_labels = self.get_data_loader().next_batch(self.batch_size, self.image_size)

    batch_filtered_gt_boxes = []
    num_outsourced = 0
    for gt_boxes, gt_landmarks in batch_gt_labels:
      filtered_gt_boxes = []
      for gt_box in gt_boxes:
        if min(gt_box[2] * self.image_size, gt_box[3] * self.image_size) > self.min_box_size_px:
          filtered_gt_boxes.append(gt_box)
        else:
          num_outsourced += 1
      batch_filtered_gt_boxes.append(filtered_gt_boxes)

    # t = time()
    masks = self.ground_truth_label_creator.create_gt_masks(batch_filtered_gt_boxes)
    self.update_epoch_stats(batch_filtered_gt_boxes, masks, num_outsourced)
    # print(time() - t)
    targets = []

    for stage_idx in range(0, len(self.anchors_by_stage)):
      for anchor_idx in range(0, len(self.anchors_by_stage[stage_idx])):
        pos_anchors_mask = masks["pos_anchors_masks_by_stage"][stage_idx][anchor_idx]
        neg_anchors_mask = masks["neg_anchors_masks_by_stage"][stage_idx][anchor_idx]

        offsets = masks["offsets_by_stage"][stage_idx][anchor_idx]
        scales = masks["scales_by_stage"][stage_idx][anchor_idx]
        targets.append(np.concatenate([offsets, pos_anchors_mask], axis=3))
        targets.append(np.concatenate([scales, pos_anchors_mask], axis=3))
        targets.append(pos_anchors_mask)
        targets.append(neg_anchors_mask)

    return np.array(batch_x), targets

  def get_data_loader(self):
    return self.train_data_loader

  def on_epoch_end(self):
    checkpoint_name = self.model_name + '_epoch' + str(self.epoch) + '.hdf5'
    trainlog_name = self.model_name + '_train_log.txt'
    self.model.save_weights(checkpoint_name)

    drive_folder_id = '13cy5wLtXwmM1XXNCR-MOq377AH3QFNaj'
    try_upload_file(checkpoint_name, drive_folder_id)
    try_upload_file(trainlog_name, drive_folder_id)

    self.epoch += 1

    self.print_epoch_stats()
    self.reset_epoch_stats()
