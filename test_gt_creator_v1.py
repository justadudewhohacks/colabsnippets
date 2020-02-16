import numpy as np

from colabsnippets.face_detection.fpn import create_anchor
from colabsnippets.face_detection.fpn.keras.GroundTruthLabelCreatorV1 import GroundTruthLabelCreatorV1

image_size = 640
anchors_by_stage = [[create_anchor(a)] for a in [256, 128, 64, 32, 16]]
stage_num_cells = [5, 10, 20, 40, 80]

gt_creator = GroundTruthLabelCreatorV1(image_size, anchors_by_stage, stage_num_cells)
gt_creator.initialize_anchors()

#abs_gt_boxes = [[40, 40, 250, 400], [40, 40, 240, 410]]
abs_gt_boxes = [[4, 6, 15, 26], [8, 5, 47, 54]]
masks = gt_creator.create_gt_masks([np.array(abs_gt_boxes) / image_size])

for s in range(0, 5):
  print('stage', s)
  for a in range(0, 1):
    print("pos_anchors_masks_by_stage")
    print(np.where(masks["pos_anchors_masks_by_stage"][s][a] > 0))
    print("neg_anchors_masks_by_stage")
    print(np.where(masks["neg_anchors_masks_by_stage"][s][a] < 1))
    print("offsets_by_stage", masks["offsets_by_stage"][s][a].shape)
    print(np.where(masks["offsets_by_stage"][s][a] != 0))
    print(masks["offsets_by_stage"][s][a][(masks["offsets_by_stage"][s][a] != 0)])
    print("scales_by_stage", masks["scales_by_stage"][s][a].shape)
    print(np.where(masks["scales_by_stage"][s][a] != 0))
    print(masks["scales_by_stage"][s][a][(masks["scales_by_stage"][s][a] != 0)])
    print("gt_boxes_with_no_matching_anchors")
    print(masks["gt_boxes_with_no_matching_anchors"])
    print("gt_boxes_with_multiple_matching_anchors")
    print(masks["gt_boxes_with_multiple_matching_anchors"])

    print("test")
    print(np.sum((masks["pos_anchors_masks_by_stage"][s][a] != 0) & (masks["pos_anchors_masks_by_stage"][s][a] != 1)))
    print(np.sum((masks["neg_anchors_masks_by_stage"][s][a] != 0) & (masks["neg_anchors_masks_by_stage"][s][a] != 1)))
