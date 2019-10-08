import random

from colabsnippets.utils import min_bbox, fix_boxes


def num_in_range(val, min_val, max_val):
  return min(max(min_val, val), max_val)


def abs_coords(bbox, hw):
  height, width = hw[:2]
  min_x, min_y, max_x_or_w, max_y_or_h = bbox
  return [int(min_x * width), int(min_y * height), int(max_x_or_w * width), int(max_y_or_h * height)]


def rel_coords(bbox, hw):
  height, width = hw[:2]
  min_x, min_y, max_x_or_w, max_y_or_h = bbox
  return [min_x / width, min_y / height, max_x_or_w / width, max_y_or_h / height]

def crop(img, boxes, crop_range=0.0, is_bbox_safe=True):
  if is_bbox_safe:
    roi = min_bbox(boxes)
  else:
    roi = random.choice(boxes) if len(boxes > 0) else None

  if roi is None:
    return img, boxes

  height, width = img.shape[:2]
  x, y, w, h = roi
  min_x, min_y, max_x, max_y = [num_in_range(v, 0, 1) for v in [x, y, x + w, y + h]]
  min_x, max_x = [int(v * width) for v in [min_x, max_x]]
  min_y, max_y = [int(v * height) for v in [min_y, max_y]]

  crop_range = num_in_range(crop_range, 0, 1)
  x0 = random.randint(round(crop_range * min_x), min_x)
  y0 = random.randint(round(crop_range * min_y), min_y)
  x1 = random.randint(0, round((1.0 - crop_range) * abs(width - max_x))) + max_x
  y1 = random.randint(0, round((1.0 - crop_range) * abs(height - max_y))) + max_y
  cropped_img = img[y0:y1, x0:x1]

  shifted_rel_boxes = []
  for box in boxes:
    x, y, w, h = abs_coords(box, img.shape)
    sx = x - x0
    sy = y - y0
    shifted_rel_boxes.append(rel_coords((sx, sy, w, h), cropped_img.shape))

  shifted_rel_boxes = fix_boxes([num_in_range(v, 0, 1) for v in shifted_rel_boxes], min(cropped_img.shape[0:2]), 1)
  return cropped_img, shifted_rel_boxes