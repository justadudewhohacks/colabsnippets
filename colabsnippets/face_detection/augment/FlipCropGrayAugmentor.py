import random

import cv2

from .AlbumentationsAugmentorBaseV1 import AlbumentationsAugmentorBaseV1
from .crop import crop


class FlipCropGrayAugmentor(AlbumentationsAugmentorBaseV1):
  def __init__(self, albumentations_lib):
    super().__init__(albumentations_lib)

    self.crop_min_box_target_size = 0.0
    self.crop_max_cutoff = 0.5
    self.crop_is_bbox_safe = False

    self.prob_crop = 0.5
    self.prob_flip = 0.5
    self.prob_gray = 0.2

  def _augment_abs_boxes(self, img, boxes, resize):
    transforms = self.albumentations_lib.augmentations.transforms
    Compose = self.albumentations_lib.Compose

    if random.random() <= self.prob_crop:
      img, boxes = crop(img, boxes, is_bbox_safe=self.crop_is_bbox_safe, max_cutoff=self.crop_max_cutoff, min_box_target_size=self.crop_min_box_target_size)
      boxes = self._fix_abs_boxes(boxes, img.shape[0:2])

    res = Compose([
      transforms.LongestMaxSize(p=1.0, max_size=resize),
      transforms.HorizontalFlip(p=self.prob_flip),
      transforms.ToGray(p=self.prob_gray),
      transforms.PadIfNeeded(p=1.0, min_height=resize, min_width=resize, border_mode=cv2.BORDER_CONSTANT)
    ], self.bbox_params)(image=img, bboxes=boxes, labels=['' for _ in boxes])

    img, boxes = res['image'], res['bboxes']
    return img, boxes
