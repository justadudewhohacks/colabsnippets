import random

from colabsnippets.utils import min_bbox, num_in_range, rel_bbox_coords, filter_abs_boxes
from .utils import filter_abs_boxes_out_of_borders, filter_abs_landmarks_out_of_borders


def crop(img, abs_boxes, abs_landmarks, crop_range=0.0, is_bbox_safe=True, min_box_target_size=0):
  im_h, im_w = img.shape[:2]
  target_rel_boxes = [rel_bbox_coords(abs_box, [im_h, im_w]) for abs_box in
                      filter_abs_boxes(abs_boxes, min_box_target_size)]
  if len(target_rel_boxes) < 1:
    return img, abs_boxes, abs_landmarks
  roi = min_bbox(target_rel_boxes) if is_bbox_safe else random.choice(target_rel_boxes)

  rx, ry, rw, rh = roi
  min_x, min_y, max_x, max_y = [num_in_range(v, 0, 1) for v in [rx, ry, rx + rw, ry + rh]]
  min_x, max_x = [int(v * im_w) for v in [min_x, max_x]]
  min_y, max_y = [int(v * im_h) for v in [min_y, max_y]]

  crop_range = num_in_range(crop_range, 0, 1)
  crop_x0 = random.randint(round(crop_range * min_x), min_x)
  crop_y0 = random.randint(round(crop_range * min_y), min_y)
  crop_x1 = random.randint(0, round((1.0 - crop_range) * abs(im_w - max_x))) + max_x
  crop_y1 = random.randint(0, round((1.0 - crop_range) * abs(im_h - max_y))) + max_y
  cropped_img = img[crop_y0:crop_y1, crop_x0:crop_x1]

  shifted_abs_boxes = [[x - crop_x0, y - crop_y0, w, h] for x, y, w, h in abs_boxes]
  filtered_abs_boxes = filter_abs_boxes_out_of_borders(shifted_abs_boxes, cropped_img.shape[0:2])
  shifted_abs_landmarks = [[(x - crop_x0, y - crop_y0) for x, y in l] for l in abs_landmarks]
  filtered_abs_landmarks = filter_abs_landmarks_out_of_borders(shifted_abs_landmarks, cropped_img.shape[0:2])

  return cropped_img, filtered_abs_boxes, filtered_abs_landmarks
