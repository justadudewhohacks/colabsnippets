import math

import numpy as np
from keras.utils import Sequence

from colabsnippets.utils import try_upload_file, load_json
from ..DataLoader import DataLoader

class TrainFPNV1(Sequence):
  def __init__(self, image_augmentor, augmentation_prob, batch_size, image_size, anchors_by_stage, stage_num_cells,
               model, epoch=0, is_wider_only=False, is_exclude_wider=False, with_mafa_and_ufdd=False,
               with_celeba_face_clusters=False, model_name='none', drive_folder_id='13cy5wLtXwmM1XXNCR-MOq377AH3QFNaj',
               start_epoch=None):
    self.batch_size = batch_size
    self.image_size = image_size
    self.anchors_by_stage = anchors_by_stage
    self.stage_num_cells = stage_num_cells
    self.model = model
    self.epoch = epoch
    self.model_name = model_name
    self.pos_iou_threshold = 0.5
    self.neg_iou_threshold = 0.3
    self.min_box_size_px = 8
    self.drive_folder_id = drive_folder_id

    self.stage_anchor_corner_coords = []
    self.stage_anchor_descriptors = []
    for stage_idx, stage_anchors in enumerate(anchors_by_stage):
      num_cells = stage_num_cells[stage_idx]
      cell_size = image_size / num_cells
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

    wider_trainData = load_json('./wider_trainData.json')
    if is_wider_only:
      train_data = wider_trainData
    else:
      ibug_challenge_data = load_json('./ibug_challenge_data.json')
      face_detection_scrapeddb_data = load_json('./face_detection_scrapeddb_data.json')
      train_data = wider_trainData + ibug_challenge_data + face_detection_scrapeddb_data

    if is_exclude_wider:
      train_data = ibug_challenge_data + face_detection_scrapeddb_data
    if with_mafa_and_ufdd:
      mafa = load_json('./MAFA_train_data.json')
      ufdd = load_json('./UFDD_val_data.json')
      for d in mafa:
        train_data.append(d)
      for d in ufdd:
        train_data.append(d)
    if with_celeba_face_clusters:
      celeba_face_clusters = load_json('./celeba_face_clusters_data.json')
      for d in celeba_face_clusters:
        train_data.append(d)

    self.train_data_loader = DataLoader(train_data, start_epoch=start_epoch, image_augmentor=image_augmentor,
                                        augmentation_prob=augmentation_prob, min_box_size_px=0)
    print('train samples:', len(train_data))

  def __len__(self):
    return int(math.ceil(self.get_data_loader().get_end_idx() / self.batch_size))

  def __getitem__(self, idx):
    batch_x, batch_gt_labels = self.get_data_loader().next_batch(self.batch_size, self.image_size)

    batch_filtered_gt_boxes = []
    for gt_boxes, gt_landmarks in batch_gt_labels:
      filtered_gt_boxes = []
      for gt_box in gt_boxes:
        if min(gt_box[2] * self.image_size, gt_box[3] * self.image_size) > self.min_box_size_px:
          filtered_gt_boxes.append(gt_box)
      batch_filtered_gt_boxes.append(filtered_gt_boxes)

    # t = time()
    masks = self.create_gt_masks(batch_filtered_gt_boxes)
    # print(time() - t)
    num_stages = len(self.anchors_by_stage)
    targets = []
    for stage_idx in range(0, num_stages):
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
    # TODO
    return self.train_data_loader
    current_epoch = self.epoch
    is_wider_anchor_epoch = not current_epoch == 0 and (current_epoch + 20) % 20 == 0
    is_ibug_anchor_epoch = (current_epoch + 10) % 20 == 0
    is_anchor_epoch = is_wider_anchor_epoch or is_ibug_anchor_epoch
    if is_anchor_epoch:
      if is_wider_anchor_epoch:
        return self.wider_anchor_epoch_data_loader
      if is_ibug_anchor_epoch:
        return self.ibug_anchor_epoch_data_loader
    return self.train_data_loader

  def on_epoch_end(self):
    checkpoint_name = self.model_name + '_epoch' + str(self.epoch) + '.hdf5'
    trainlog_name = self.model_name + '_train_log.txt'
    self.model.save_weights(checkpoint_name)

    drive_folder_id = self.drive_folder_id
    try_upload_file(checkpoint_name, drive_folder_id)
    try_upload_file(trainlog_name, drive_folder_id)

    self.epoch += 1