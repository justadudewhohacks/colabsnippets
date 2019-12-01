import random

import cv2

from colabsnippets.face_detection.augment.anchor_based_sampling import anchor_based_sampling
from colabsnippets.face_detection.augment.random_pad_to_square import random_pad_to_square
from colabsnippets.utils import rel_bbox_coords, min_bbox
from .AlbumentationsAugmentorBase import AlbumentationsAugmentorBase
from .crop import crop


class AlbumentationsAugmentor(AlbumentationsAugmentorBase):
  def __init__(self, albumentations_lib):
    super().__init__(albumentations_lib)

    self.prob_rotate = 0.5
    self.prob_stretch = 0.25
    self.max_rotation_angle = 30
    self.max_stretch_x = 1.4
    self.max_stretch_y = 1.4

    self.crop_min_box_target_size = 0.0
    self.crop_max_cutoff = 0.5
    self.crop_is_bbox_safe = False

    self.prob_anchor_based_sampling = 0.5
    self.anchor_based_sampling_anchors = [16, 32, 64, 128, 256]
    self.anchor_based_sampling_max_scale = 4.0

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
    self.prob_blur = 0.25
    self.prob_dropout = 0.25

    self.debug = False

  # assuming image is square with (size, size)
  def _get_stretch_shape(self, size):
    is_stretch_x = random.choice([True, False])

    stretch_factor = self.max_stretch_x if is_stretch_x else self.max_stretch_y
    stretched_dim = random.uniform(1.0, stretch_factor) * size
    scale_f = size / stretched_dim
    down_scaled_dim = int(round(scale_f * size))

    return (size, down_scaled_dim) if is_stretch_x else (down_scaled_dim, size)

  def _augment_abs_boxes(self, img, boxes, resize):
    transforms = self.albumentations_lib.augmentations.transforms
    Compose = self.albumentations_lib.Compose

    if random.random() <= self.prob_crop:
      if self.debug:
        print('applying crop')
      img, boxes = crop(img, boxes, is_bbox_safe=self.crop_is_bbox_safe, max_cutoff=self.crop_max_cutoff,
                        min_box_target_size=self.crop_min_box_target_size)
      if self.debug:
        print('boxes after crop:', boxes)
      boxes = self._fix_abs_boxes(boxes, img.shape[0:2])
      if self.debug:
        print('filtering with dimensions:', str(img.shape[0:2]))
        print('boxes after filtering:', boxes)

    # pre downscale
    if self.debug:
      print('applying pre downscale')
    pre_downscale_size = 1.5 * resize
    aug_pre_downscale = Compose([
      transforms.LongestMaxSize(p=1.0, max_size=pre_downscale_size)
    ], self.bbox_params)
    if max(img.shape[0:2]) > pre_downscale_size:
      res = aug_pre_downscale(image=img, bboxes=boxes, labels=['' for _ in boxes])
      img, boxes = res['image'], res['bboxes']

    if random.random() < self.prob_rotate:
      if self.debug:
        print('applying rotation')
      aug_rot = Compose([
        # pad borders to avoid cutting out parts of image when rotating it
        transforms.PadIfNeeded(p=1.0, min_height=int(pre_downscale_size), min_width=int(pre_downscale_size),
                               border_mode=cv2.BORDER_CONSTANT),
        transforms.Rotate(p=1.0, limit=(-self.max_rotation_angle, self.max_rotation_angle),
                          border_mode=cv2.BORDER_CONSTANT)
      ], self.bbox_params)
      res = aug_rot(image=img, bboxes=boxes, labels=['' for _ in boxes])
      img, boxes = res['image'], res['bboxes']
      boxes = self._fix_abs_boxes(boxes, img.shape[0:2])

    if random.random() <= self.prob_anchor_based_sampling and self.anchor_based_sampling_anchors is not None:
      if self.debug:
        print('applying anchor_based_sampling')
      img, boxes = anchor_based_sampling(img, boxes, self.anchor_based_sampling_anchors,
                                         max_scale=self.anchor_based_sampling_max_scale)
      boxes = self._fix_abs_boxes(boxes, img.shape[0:2])

    # crop to max output size
    if self.debug:
      print('applying crop to max output size')
    im_h, im_w = img.shape[0:2]
    rx, ry, rw, rh = min_bbox(rel_bbox_coords(abs_box, [im_h, im_w]) for abs_box in boxes)
    rcx, rcy = [(rx + (rw / 2)) * im_w, (ry + (rh / 2)) * im_h]
    crop_x0 = int(max(0, rcx - (resize / 2)))
    crop_y0 = int(max(0, rcy - (resize / 2)))
    crop_x1 = int(min(im_w, rcx + (resize / 2)))
    crop_y1 = int(min(im_h, rcy + (resize / 2)))
    img = img[crop_y0:crop_y1, crop_x0:crop_x1, :]
    boxes = self._fix_abs_boxes([(x - crop_x0, y - crop_y0, w, h) for x, y, w, h in boxes], img.shape[0:2])

    if self.debug:
      print('applying random_pad_to_square')
    img, boxes = random_pad_to_square(img, boxes, resize)

    if self.debug:
      print('applying transformations')
      print(boxes)
    stretch_x, stretch_y = self._get_stretch_shape(resize)
    res = Compose([
      transforms.HorizontalFlip(p=self.prob_flip),
      transforms.Resize(stretch_y, stretch_x, p=self.prob_stretch)
    ], self.bbox_params)(image=img, bboxes=boxes, labels=['' for _ in boxes])
    img, boxes = res['image'], res['bboxes']

    if self.debug:
      print('applying color distortion')
    # TODO: shear, rescale blur?
    transformations = [
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
    ]
    img = Compose(transformations)(image=img)['image']
    # CUTOUT
    # img = transforms.Cutout(p = 1.0, num_holes=8, max_h_size=8, max_w_size=8).apply(img)

    # DISTORTIONS
    # TODO: a bit slow
    # img = transforms.OpticalDistortion(p = 1.0, distort_limit=0.05, shift_limit=0.05).apply(img)
    # TODO GridDistortion?
    # img = transforms.GridDistortion(p = 1.0, num_steps=5, distort_limit=0.3).apply(img)
    # TODO: very slow
    # img = transforms.ElasticTransform(p = 1.0, alpha=1, sigma=50, alpha_affine=50).apply(img)
    # TODO: module 'albumentations.augmentations.transforms' has no attribute 'RandomGridShuffle'
    # img = transforms.RandomGridShuffle(p = 1.0, grid=(3, 3)).apply(img)

    return img, boxes
