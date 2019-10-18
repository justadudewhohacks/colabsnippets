import random

from colabsnippets.utils import min_bbox, num_in_range, rel_bbox_coords, filter_abs_boxes


def crop(img, abs_boxes, crop_range=0.0, is_bbox_safe=True, max_cutoff=0.5, min_box_target_size=0):
  im_h, im_w = img.shape[:2]
  target_rel_boxes = [rel_bbox_coords(abs_box, [im_h, im_w]) for abs_box in
                      filter_abs_boxes(abs_boxes, min_box_target_size)]
  if len(target_rel_boxes) < 1:
    return img, abs_boxes
  roi = min_bbox(target_rel_boxes) if is_bbox_safe else random.choice(target_rel_boxes)

  rx, ry, rw, rh = roi
  min_x, min_y, max_x, max_y = [num_in_range(v, 0, 1) for v in [rx, ry, rx + rw, ry + rh]]
  min_x, max_x = [int(v * im_w) for v in [min_x, max_x]]
  min_y, max_y = [int(v * im_h) for v in [min_y, max_y]]

  crop_range = num_in_range(crop_range, 0, 1)
  x0 = random.randint(round(crop_range * min_x), min_x)
  y0 = random.randint(round(crop_range * min_y), min_y)
  x1 = random.randint(0, round((1.0 - crop_range) * abs(im_w - max_x))) + max_x
  y1 = random.randint(0, round((1.0 - crop_range) * abs(im_h - max_y))) + max_y
  cropped_img = img[y0:y1, x0:x1]

  shifted_rel_boxes = []
  for abs_box in abs_boxes:
    x, y, w, h = abs_box
    sx = x - x0
    sy = y - y0
    shifted_rel_boxes.append(rel_bbox_coords((sx, sy, w, h), cropped_img.shape))

  filtered_boxes = []
  for box in shifted_rel_boxes:
    x0, y0, w, h = box
    x0, y0, x1, y1 = [num_in_range(v, 0, 1) for v in [x0, y0, x0 + w, y0 + h]]
    w_cut = max(0, x1 - x0)
    h_cut = max(0, y1 - y0)
    area_prev = w * h
    area_cut = w_cut * h_cut
    ratio = area_cut / area_prev

    if ratio >= max_cutoff:
      filtered_boxes.append(box)

  return cropped_img, filtered_boxes
