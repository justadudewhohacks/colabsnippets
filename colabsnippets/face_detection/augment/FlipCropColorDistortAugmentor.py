import random

import cv2

from .AlbumentationsAugmentorBase import AlbumentationsAugmentorBase
from .crop import crop


class FlipCropColorDistortAugmentor(AlbumentationsAugmentorBase):
  def __init__(self, albumentations_lib):
    super().__init__(albumentations_lib)

    self.crop_min_box_target_size = 0.0
    self.crop_max_cutoff = 0.5

    self.prob_crop = 0.5
    self.prob_flip = 0.5
    self.prob_gray = 0.2

    self.gamma_limit = (80, 120)
    self.hue_shift_limit = 20
    self.sat_shift_limit = 30
    self.val_shift_limit = 20
    self.r_shift_limit = 20
    self.g_shift_limit = 20
    self.b_shift_limit = 20
    self.brightness_limit = 0.2
    self.contrast_limit = 0.2
    self.blur_multiplier = 1.0
    self.max_holes = 16
    self.max_hole_rel_size = 0.05

    self.prob_gamma = 0.5
    self.prob_hsv = 0.5
    self.prob_rgb = 0.5
    self.prob_brightness_contrast = 0.5
    self.prob_gray = 0.2
    self.prob_blur = 0.25
    self.prob_dropout = 0.25

  def _augment_abs_boxes(self, img, boxes, resize):
    transforms = self.albumentations_lib.augmentations.transforms
    Compose = self.albumentations_lib.Compose

    if random.random() <= self.prob_crop:
      img, boxes = crop(img, boxes, max_cutoff=self.crop_max_cutoff, min_box_target_size=self.crop_min_box_target_size)
      boxes = self._fix_abs_boxes(boxes, img.shape[0:2])

    res = Compose([
      transforms.LongestMaxSize(p=1.0, max_size=resize),
      transforms.HorizontalFlip(p=self.prob_flip),
      transforms.PadIfNeeded(p=1.0, min_height=resize, min_width=resize, border_mode=cv2.BORDER_CONSTANT)
    ], self.bbox_params)(image=img, bboxes=boxes, labels=['' for _ in boxes])
    img, boxes = res['image'], res['bboxes']

    img = Compose([
      transforms.ToGray(p=self.prob_gray),
      transforms.RandomGamma(p=self.prob_gamma, gamma_limit=self.gamma_limit),
      transforms.HueSaturationValue(p=self.prob_hsv, hue_shift_limit=self.hue_shift_limit,
                                    sat_shift_limit=self.sat_shift_limit, val_shift_limit=self.val_shift_limit),
      transforms.RGBShift(p=self.prob_rgb, r_shift_limit=self.r_shift_limit, g_shift_limit=self.g_shift_limit,
                          b_shift_limit=self.b_shift_limit),
      transforms.RandomBrightnessContrast(p=self.prob_brightness_contrast, brightness_limit=self.brightness_limit,
                                          contrast_limit=self.contrast_limit),
      transforms.Blur(p=self.prob_blur, blur_limit=int((self.blur_multiplier * resize) / 100)),
      transforms.CoarseDropout(p=self.prob_dropout, max_holes=self.max_holes,
                               max_height=int(self.max_hole_rel_size * resize),
                               max_width=int(self.max_hole_rel_size * resize), fill_value=random.randint(0, 255))
    ])(image=img)['image']

    return img, boxes
