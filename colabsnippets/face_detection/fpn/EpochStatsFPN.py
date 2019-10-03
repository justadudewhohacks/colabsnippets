import time
import numpy as np
from ...utils import flatten_list
from .calculate_stats import calculate_stats
from .FPNBase import batch_boxes_by_stage_to_boxes_by_batch

class EpochStatsFPN:
  def __init__(self, num_stages = 3):
    self.num_stages = num_stages
    self.total_object_losses_by_stage = np.zeros(self.num_stages)
    self.total_offsets_losses_by_stage = np.zeros(self.num_stages)
    self.total_scales_losses_by_stage = np.zeros(self.num_stages)
    self.total_no_object_losses_by_stage = np.zeros(self.num_stages)
    self.num_gt_anchors_by_stage = np.zeros(self.num_stages)
    self.num_tps_score_by_stage = np.zeros(self.num_stages)
    self.num_preds_by_stage = np.zeros(self.num_stages)

    self.total_loss = 0
    self.total_tps_01 = 0
    self.total_matches_01 = 0
    self.total_tps_03 = 0
    self.total_matches_03 = 0
    self.total_tps_05 = 0
    self.total_matches_05 = 0
    self.total_num_preds = 0
    self.total_num_gt_boxes = 0
    self.ts_epoch = time.time()

  def update(self, preds, batch_gt_boxes):
    inputs = {
      "batch_gt_boxes": batch_gt_boxes,
      "batch_scores_by_stage": preds["batch_scores_by_stage"],
      # TODO
      "batch_pred_boxes": batch_boxes_by_stage_to_boxes_by_batch(preds["batch_pred_boxes_by_stage"]),
      "pos_anchors_masks_by_stage": preds["pos_anchors_masks_by_stage"]
    }
    # TODO: by stage
    stats = calculate_stats(inputs, score_thresh = 0.5, iou_threshs = [0.1, 0.3, 0.5])

    self.total_object_losses_by_stage += np.array(preds["weighted_object_losses_by_stage"])
    self.total_offsets_losses_by_stage += np.array(preds["weighted_no_object_losses_by_stage"])
    self.total_scales_losses_by_stage += np.array(preds["weighted_offset_losses_by_stage"])
    self.total_no_object_losses_by_stage += np.array(preds["weighted_scales_losses_by_stage"])
    self.num_gt_anchors_by_stage += np.array(stats["num_gt_anchors_by_stage"])
    self.num_tps_score_by_stage += stats["num_tps_score_by_stage"]
    self.num_preds_by_stage += np.array([len(flatten_list(batch_preds)) for batch_preds in preds["batch_pred_boxes_by_stage"]])

    self.total_loss += preds["loss"]
    self.total_tps_01 += stats["num_tps"][0]
    self.total_matches_01 += stats["num_matches"][0]
    self.total_tps_03 += stats["num_tps"][1]
    self.total_matches_03 += stats["num_matches"][1]
    self.total_tps_05 += stats["num_tps"][2]
    self.total_matches_05 += stats["num_matches"][2]
    self.total_num_preds += stats["num_preds"]
    self.total_num_gt_boxes += stats["num_gt_boxes"]

    return stats

  def write_stats(self, epoch_txt):
    iteration_count = np.sum(self.num_gt_anchors_by_stage)
    total_tps_score = np.sum(self.num_tps_score_by_stage)

    # TODO: gt anchors by anchor
    # tps/fps by anchor

    avg_loss = self.total_loss / iteration_count
    avg_object_losses_by_stage = self.total_object_losses_by_stage / self.num_gt_anchors_by_stage
    avg_no_object_losses_by_stage = self.total_no_object_losses_by_stage / self.num_gt_anchors_by_stage
    avg_offsets_losses_by_stage = self.total_offsets_losses_by_stage / self.num_gt_anchors_by_stage
    avg_scales_losses_by_stage = self.total_scales_losses_by_stage / self.num_gt_anchors_by_stage
    avg_object_loss = np.sum(self.total_object_losses_by_stage) / iteration_count
    avg_no_object_loss = np.sum(self.total_no_object_losses_by_stage) / iteration_count
    avg_offsets_loss = np.sum(self.total_offsets_losses_by_stage) / iteration_count
    avg_scales_loss = np.sum(self.total_scales_losses_by_stage) / iteration_count

    avg_tps_score_by_stage = self.num_tps_score_by_stage / self.num_gt_anchors_by_stage
    avg_fps_score_by_stage = (self.num_preds_by_stage - self.num_tps_score_by_stage) / self.num_gt_anchors_by_stage
    avg_tps_score = total_tps_score / iteration_count
    avg_fps_score = (self.total_num_preds - total_tps_score) / iteration_count

    avg_tps_01 = self.total_tps_01 / self.total_num_gt_boxes
    avg_fps_01 = (self.total_num_preds - self.total_matches_01) / self.total_num_gt_boxes
    avg_tps_matches_01 = self.total_matches_01 / iteration_count
    avg_fps_matches_01 = (self.total_num_preds - self.total_matches_01) / iteration_count

    avg_tps_03 = self.total_tps_03 / self.total_num_gt_boxes
    avg_fps_03 = (self.total_num_preds - self.total_matches_03) / self.total_num_gt_boxes
    avg_tps_matches_03 = self.total_matches_03 / iteration_count
    avg_fps_matches_03 = (self.total_num_preds - self.total_matches_03) / iteration_count

    avg_tps_05 = self.total_tps_05 / self.total_num_gt_boxes
    avg_fps_05 = (self.total_num_preds - self.total_matches_05) / self.total_num_gt_boxes
    avg_tps_matches_05 = self.total_matches_05 / iteration_count
    avg_fps_matches_05 = (self.total_num_preds - self.total_matches_05) / iteration_count

    self.total_num_preds = self.total_num_preds if self.total_num_preds > 0 else 1
    match_to_pred_ratio_01 = self.total_matches_01 / self.total_num_preds
    match_to_pred_ratio_03 = self.total_matches_03 / self.total_num_preds
    match_to_pred_ratio_05 = self.total_matches_05 / self.total_num_preds

    def log(line):
      print(line)
      epoch_txt.write(line + '\n')

    format_array = lambda arr: str(["{:.4f}".format(l) for l in arr])

    log("----------------------------")
    log("epoch_time= {}".format(time.time() - self.ts_epoch))
    log("total_num_gt_boxes= {}".format(self.total_num_gt_boxes))
    log("----------------------------")
    log("total_num_gt_anchors= {}".format(iteration_count))
    log("total_num_preds= {}".format(self.total_num_preds))
    log("total_loss= {}".format(self.total_loss))
    log("avg_loss= {:.4f}".format(avg_loss))
    log("avg_object_loss= {:.4f}".format(avg_object_loss))
    log("avg_no_object_loss= {:.4f}".format(avg_no_object_loss))
    log("avg_offsets_loss= {:.4f}".format(avg_offsets_loss))
    log("avg_scales_loss= {:.4f}".format(avg_scales_loss))
    log("----------------------------")
    log("num_gt_anchors_by_stage= {}".format(format_array(self.num_gt_anchors_by_stage)))
    log("num_preds_by_stage= {}".format(format_array(self.num_preds_by_stage)))
    log("avg_object_losses_by_stage= {}".format(format_array(avg_object_losses_by_stage)))
    log("avg_no_object_losses_by_stage= {}".format(format_array(avg_no_object_losses_by_stage)))
    log("avg_offsets_losses_by_stage= {}".format(format_array(avg_offsets_losses_by_stage)))
    log("avg_scales_losses_by_stage= {}".format(format_array(avg_scales_losses_by_stage)))
    log("----------------------------")
    log("avg_tps_score= {:.4f}".format(avg_tps_score))
    log("avg_fps_score= {:.4f}".format(avg_fps_score))
    log("avg_tps_score_by_stage= {}".format(format_array(avg_tps_score_by_stage)))
    log("avg_fps_score_by_stage= {}".format(format_array(avg_fps_score_by_stage)))
    log("----------------------------")
    log("avg_tps_01= {:.4f}".format(avg_tps_01))
    log("avg_fps_01= {:.4f}".format(avg_fps_01))
    log("avg_tps_matches_01= {:.4f}".format(avg_tps_matches_01))
    log("avg_fps_matches_01= {:.4f}".format(avg_fps_matches_01))
    log("----------------------------")
    log("avg_tps_03= {:.4f}".format(avg_tps_03))
    log("avg_fps_03= {:.4f}".format(avg_fps_03))
    log("avg_tps_matches_03= {:.4f}".format(avg_tps_matches_03))
    log("avg_fps_matches_03= {:.4f}".format(avg_fps_matches_03))
    log("----------------------------")
    log("avg_tps_05= {:.4f}".format(avg_tps_05))
    log("avg_fps_05= {:.4f}".format(avg_fps_matches_05))
    log("avg_tps_matches_05= {:.4f}".format(avg_tps_matches_05))
    log("avg_fps_matches_05= {:.4f}".format(avg_fps_matches_05))
    log("----------------------------")
    log("match_to_pred_ratio_01= {:.4f}".format(match_to_pred_ratio_01))
    log("match_to_pred_ratio_03= {:.4f}".format(match_to_pred_ratio_03))
    log("match_to_pred_ratio_05= {:.4f}".format(match_to_pred_ratio_05))
    log("")