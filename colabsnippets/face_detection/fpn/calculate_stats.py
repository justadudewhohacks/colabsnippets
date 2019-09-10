import numpy as np

from ..calculate_iou import calculate_iou
from ..object_detector.calculate_stats import calculate_stats as _calculate_stats

def calculate_stats(inputs, score_thresh = 0.5):
  num_stages = len(inputs["batch_scores_by_stage"])

  gt_pos_by_stage = [np.where(inputs["gt_masks_by_stage"][s] != 0) for s in range(0, num_stages)]
  num_gt_anchors = np.sum([gt_pos_by_stage[s][0].shape[0] for s in range(0, num_stages)])
  num_tps_score = np.sum([np.where(inputs["batch_scores_by_stage"][s][gt_pos_by_stage[s]] > score_thresh)[0].shape[0] for s in range(0, num_stages)])

  stats = _calculate_stats(inputs, score_thresh = score_thresh)
  stats["num_tps_score"] = num_tps_score
  stats["num_gt_anchors"] = num_gt_anchors
  return stats