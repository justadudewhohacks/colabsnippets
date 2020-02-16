import numpy as np

def area_of(left_top, right_bottom):
  hw = np.clip(right_bottom - left_top, 0.0, None)
  return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
  overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
  overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

  overlap_area = area_of(overlap_left_top, overlap_right_bottom)
  area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
  area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
  return overlap_area / (area0 + area1 - overlap_area + eps)


def box_xywh_to_corner(box):
  x, y, w, h = box
  return [x, y, x + w, y + h]


def box_corner_to_xywh(box):
  x0, y0, x1, y1 = box
  return [x0, y0, x1 - x0, y1 - y0]


