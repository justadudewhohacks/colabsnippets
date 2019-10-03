import random

import cv2

from ..utils import fix_boxes


class AlbumentationsAugmentorBase:
  def __init__(self, albumentations_lib):
    self.albumentations_lib = albumentations_lib
    self.bbox_params = self.albumentations_lib.BboxParams(format='coco', label_fields=['labels'], min_area=0.0,
                                                          min_visibility=0.0)

  def _fix_rel_boxes(self, boxes):
    fixed_boxes = []
    for x, y, w, h in boxes:
      x, y = max(0, x), max(0, y)
      w, h = min(1.0 - x, w), min(1.0 - y, h)
      fixed_boxes.append((x, y, w, h))
    return fixed_boxes

  def _abs_coords(self, bbox, img):
    height, width = img.shape[:2]
    min_x, min_y, max_x_or_w, max_y_or_h = bbox
    return [int(min_x * width), int(min_y * height), int(max_x_or_w * width), int(max_y_or_h * height)]

  def _rel_coords(self, bbox, img):
    height, width = img.shape[:2]
    min_x, min_y, max_x_or_w, max_y_or_h = bbox
    return [min_x / width, min_y / height, max_x_or_w / width, max_y_or_h / height]

  def _augment_abs_boxes(self, img, boxes, resize):
    raise Exception("AlbumentationsAugmentorBase - _augment_abs_boxes not implemented")

  def augment(self, img, boxes=[], image_size=None):
    boxes = fix_boxes([self._abs_coords(box, img) for box in self._fix_rel_boxes(boxes)], max(img.shape[0:2]), 1)
    img, boxes = self._augment_abs_boxes(img, boxes, image_size)
    boxes = [self._rel_coords(box, img) for box in boxes]
    return img, boxes

  def resize_and_to_square(self, img, boxes=[], image_size=None):
    boxes = fix_boxes([self._abs_coords(box, img) for box in self._fix_rel_boxes(boxes)], max(img.shape[0:2]), 1)
    transforms = self.albumentations_lib.augmentations.transforms
    Compose = self.albumentations_lib.Compose
    res = Compose([
      transforms.LongestMaxSize(p=1.0, max_size=image_size),
      transforms.PadIfNeeded(p=1.0, min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT)
    ], self.bbox_params)(image=img, bboxes=boxes, labels=['' for _ in boxes])
    img, boxes = res['image'], res['bboxes']
    boxes = [self._rel_coords(box, img) for box in boxes]
    return img, boxes
