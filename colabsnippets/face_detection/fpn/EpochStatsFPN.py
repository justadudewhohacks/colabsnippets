import time
from .calculate_stats import calculate_stats

class EpochStatsFPN:
  def __init__(self):
    self.total_loss = 0
    self.total_loss = 0
    self.total_object_loss = 0
    self.total_offsets_loss = 0
    self.total_scales_loss = 0
    self.total_no_object_loss = 0
    self.total_tps_01 = 0
    self.total_matches_01 = 0
    self.total_tps_03 = 0
    self.total_matches_03 = 0
    self.total_tps_05 = 0
    self.total_matches_05 = 0
    self.total_tps_score = 0
    self.total_preds = 0
    self.total_num_gt_boxes = 0
    self.iteration_count = 0
    self.ts_epoch = time.time()

  def update(self, preds, batch_gt_boxes):
    inputs = {
      "batch_gt_boxes": batch_gt_boxes,
      "batch_scores_by_stage": preds["batch_scores_by_stage"],
      "batch_pred_boxes": preds["batch_pred_boxes"],
      "gt_masks_by_stage": preds["gt_masks_by_stage"]
    }
    stats = calculate_stats(inputs, score_thresh = 0.5, iou_threshs = [0.1, 0.3, 0.5])

    self.total_loss += preds["loss"]
    self.total_object_loss += preds["object_loss"]
    self.total_no_object_loss += preds["no_object_loss"]
    self.total_offsets_loss += preds["offsets_loss"]
    self.total_scales_loss += preds["scales_loss"]

    self.total_tps_01 += stats["num_tps"][0]
    self.total_matches_01 += stats["num_matches"][0]
    self.total_tps_03 += stats["num_tps"][1]
    self.total_matches_03 += stats["num_matches"][1]
    self.total_tps_05 += stats["num_tps"][2]
    self.total_matches_05 += stats["num_matches"][2]
    self.total_tps_score += stats["num_tps_score"]
    self.total_preds += stats["num_preds"]
    self.total_num_gt_boxes += stats["num_gt_boxes"]
    self.iteration_count += stats["num_gt_anchors"]

    return stats

  def write_stats(self, epoch_txt):
    avg_loss = self.total_loss / self.iteration_count
    avg_object_loss = self.total_object_loss / self.iteration_count
    avg_offsets_loss = self.total_offsets_loss / self.iteration_count
    avg_scales_loss = self.total_scales_loss / self.iteration_count
    avg_no_object_loss = self.total_no_object_loss / self.iteration_count
    avg_tps_score = self.total_tps_score / self.iteration_count
    avg_fps_score = (self.total_preds - self.total_tps_score) / self.iteration_count

    avg_tps_01 = self.total_tps_01 / self.total_num_gt_boxes
    avg_fps_01 = (self.total_preds - self.total_matches_01) / self.total_num_gt_boxes
    avg_tps_matches_01 = self.total_matches_01 / self.iteration_count
    avg_fps_matches_01 = (self.total_preds - self.total_matches_01) / self.iteration_count

    avg_tps_03 = self.total_tps_03 / self.total_num_gt_boxes
    avg_fps_03 = (self.total_preds - self.total_matches_03) / self.total_num_gt_boxes
    avg_tps_matches_03 = self.total_matches_03 / self.iteration_count
    avg_fps_matches_03 = (self.total_preds - self.total_matches_03) / self.iteration_count

    avg_tps_05 = self.total_tps_05 / self.total_num_gt_boxes
    avg_fps_05 = (self.total_preds - self.total_matches_05) / self.total_num_gt_boxes
    avg_tps_matches_05 = self.total_matches_05 / self.iteration_count
    avg_fps_matches_05 = (self.total_preds - self.total_matches_05) / self.iteration_count

    match_to_pred_ratio_01 = self.total_matches_01 / self.total_preds
    match_to_pred_ratio_03 = self.total_matches_03 / self.total_preds
    match_to_pred_ratio_05 = self.total_matches_05 / self.total_preds

    epoch_txt.write('----------------------------\n')
    epoch_txt.write('epoch_time= ' + str(time.time() - self.ts_epoch) + 's \n')
    epoch_txt.write('total_num_gt_boxes= ' + str(self.total_num_gt_boxes) + '\n')
    epoch_txt.write('total_num_gt_anchors= ' + str(self.iteration_count) + '\n')
    epoch_txt.write('total_loss= ' + str(self.total_loss) + '\n')
    epoch_txt.write('avg_loss= ' + str(avg_loss) + '\n')
    epoch_txt.write('avg_object_loss= ' + str(avg_object_loss) + '\n')
    epoch_txt.write('avg_no_object_loss= ' + str(avg_no_object_loss) + '\n')
    epoch_txt.write('avg_offsets_loss= ' + str(avg_offsets_loss) + '\n')
    epoch_txt.write('avg_scales_loss= ' + str(avg_scales_loss) + '\n')
    epoch_txt.write('----------------------------\n')
    epoch_txt.write('avg_tps_score= ' + str(avg_tps_score) + '\n')
    epoch_txt.write('avg_fps_score= ' + str(avg_fps_score) + '\n')
    epoch_txt.write('----------------------------\n')
    epoch_txt.write('avg_tps_01= ' + str(avg_tps_01) + '\n')
    epoch_txt.write('avg_fps_01= ' + str(avg_fps_01) + '\n')
    epoch_txt.write('avg_tps_matches_01= ' + str(avg_tps_matches_01) + '\n')
    epoch_txt.write('avg_fps_matches_01= ' + str(avg_fps_matches_01) + '\n')
    epoch_txt.write('----------------------------\n')
    epoch_txt.write('avg_tps_03= ' + str(avg_tps_03) + '\n')
    epoch_txt.write('avg_fps_03= ' + str(avg_fps_03) + '\n')
    epoch_txt.write('avg_tps_matches_03= ' + str(avg_tps_matches_03) + '\n')
    epoch_txt.write('avg_fps_matches_03= ' + str(avg_fps_matches_03) + '\n')
    epoch_txt.write('----------------------------\n')
    epoch_txt.write('avg_tps_05= ' + str(avg_tps_05) + '\n')
    epoch_txt.write('avg_fps_05= ' + str(avg_fps_matches_05) + '\n')
    epoch_txt.write('avg_tps_matches_05= ' + str(avg_tps_matches_05) + '\n')
    epoch_txt.write('avg_fps_matches_05= ' + str(avg_fps_matches_05) + '\n')
    epoch_txt.write('----------------------------\n')
    epoch_txt.write('match_to_pred_ratio_01= ' + str(match_to_pred_ratio_01) + '\n')
    epoch_txt.write('match_to_pred_ratio_03= ' + str(match_to_pred_ratio_03) + '\n')
    epoch_txt.write('match_to_pred_ratio_05= ' + str(match_to_pred_ratio_05) + '\n')

    print('----------------------------')
    print('avg_loss= ' + str(avg_loss))
    print('avg_object_loss= ' + str(avg_object_loss))
    print('avg_offsets_loss= ' + str(avg_offsets_loss))
    print('avg_scales_loss= ' + str(avg_scales_loss))
    print('avg_no_object_loss= ' + str(avg_no_object_loss))
    print('total_num_gt_boxes= ' + str(self.total_num_gt_boxes))
    print('total_num_gt_anchors= ' + str(self.iteration_count))
    print('----------------------------')
    print('avg_tps_01= ' + str(avg_tps_01))
    print('match_to_pred_ratio_01= ' + str(match_to_pred_ratio_01))
    print('avg_tps_03= ' + str(avg_tps_03))
    print('match_to_pred_ratio_03= ' + str(match_to_pred_ratio_03))
    print('avg_tps_05= ' + str(avg_tps_05))
    print('match_to_pred_ratio_05= ' + str(match_to_pred_ratio_05))
    print()