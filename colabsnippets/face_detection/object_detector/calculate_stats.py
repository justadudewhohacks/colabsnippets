import numpy as np

from ..calculate_iou import calculate_iou

def calculate_stats(inputs, score_thresh = 0.5, iou_threshs = [0.5]):
  batch_size = len(inputs["batch_pred_boxes"])

  num_gt_boxes = 0
  num_preds = 0
  num_tps = [0 for i in range (0, len(iou_threshs))]
  num_matches = [0 for i in range (0, len(iou_threshs))]
  batch_good_boxes = [[[] for b in range(0, batch_size)] for i in range (0, len(iou_threshs))]
  for b in range(0, batch_size):
    gt_boxes = inputs["batch_gt_boxes"][b]
    pred_boxes = inputs["batch_pred_boxes"][b]
    num_preds += len(pred_boxes)
    num_gt_boxes += len(gt_boxes)
    for gt_box in gt_boxes:
      ious = [calculate_iou(np.array(gt_box), np.array(pred_box)) for pred_box in pred_boxes]

      for iou_idx in range (0, len(iou_threshs)):
        iou_thresh = iou_threshs[iou_idx]

        for i, iou in enumerate(ious):
          if iou > iou_thresh:
            batch_good_boxes[iou_idx][b].append(pred_boxes[i])

        for iou in ious:
          if iou > iou_thresh:
            num_tps[iou_idx] += 1
            break

    for iou_idx in range (0, len(iou_threshs)):
      iou_thresh = iou_threshs[iou_idx]
      for pred_box in pred_boxes:
        for gt_box in gt_boxes:
          iou = calculate_iou(np.array(gt_box), np.array(pred_box))
          if iou > iou_thresh:
            num_matches[iou_idx] += 1
            break

  return {
    "num_tps": num_tps,
    "num_matches": num_matches,
    "num_preds": num_preds,
    "num_gt_boxes": num_gt_boxes,
    "batch_good_boxes": batch_good_boxes
  }