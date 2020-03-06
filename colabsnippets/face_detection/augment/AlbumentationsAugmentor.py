import math
import random

import cv2

from .AlbumentationsAugmentorBase import AlbumentationsAugmentorBase, resize_to_max
from .anchor_based_sampling import anchor_based_sampling
from .crop import crop
from .resize_by_ratio import resize_by_ratio


def landmarks_to_keypoints(all_landmarks):
  keypoints = []
  for landmarks in all_landmarks:
    for x, y in landmarks:
      keypoints.append([x, y])
  return keypoints


def keypoints_to_landmarks(all_keypoints):
  if len(all_keypoints) % 5 != 0:
    raise Exception(
      'keypoints_to_landmarks - expected length of all_keypoints to be a multiple of 5, but have length: ' + str(
        len(all_keypoints)))
  num_landmarks = int(len(all_keypoints) / 5)
  return [[keypoint for keypoint in all_keypoints[i * 5:i * 5 + 5]] for i in range(0, num_landmarks)]


def pack_abs_boxes_and_abs_landmarks(abs_boxes, abs_landmarks):
  keypoints = []
  for x0, y0, w, h in abs_boxes:
    x1, y1 = x0 + w, y0 + h
    for corner in [(x0, y0), (x1, y1), (x0, y1), (x1, y0)]:
      keypoints.append(corner)
  for kp in landmarks_to_keypoints(abs_landmarks):
    keypoints.append(kp)
  return keypoints


def unpack_abs_boxes_and_abs_landmarks(keypoints, num_boxes, num_landmarks):
  box_keypoints = keypoints[0:4 * num_boxes]
  landmark_keypoints = keypoints[4 * num_boxes:]

  if len(keypoints) != (num_boxes * 4 + num_landmarks * 5):
    raise Exception("unpack_abs_boxes_and_abs_landmarks - mismatch between number of keypoints (" + str(len(keypoints))
                    + ") and num_boxes (" + str(num_boxes) + ") and num_landmarks (" + str(num_landmarks) + ")")

  abs_boxes = []
  for i in range(0, num_boxes):
    xmin, ymin, xmax, ymax = math.inf, math.inf, 0, 0
    corners = box_keypoints[4 * i:4 * i + 4]
    for x, y in corners:
      xmin, ymin = min(x, xmin), min(y, ymin)
      xmax, ymax = max(x, xmax), max(y, ymax)
    abs_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
  abs_landmarks = keypoints_to_landmarks(landmark_keypoints)
  return abs_boxes, abs_landmarks


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

    self.with_pre_downscale = True
    self.debug = False

  # assuming image is square with (size, size)
  def _get_stretch_shape(self, size):
    is_stretch_x = random.choice([True, False])

    stretch_factor = self.max_stretch_x if is_stretch_x else self.max_stretch_y
    stretched_dim = random.uniform(1.0, stretch_factor) * size
    scale_f = size / stretched_dim
    down_scaled_dim = int(round(scale_f * size))

    return (size, down_scaled_dim) if is_stretch_x else (down_scaled_dim, size)

  def _augment_abs_boxes(self, img, boxes, landmarks, resize, return_augmentation_history=False):
    augmentation_history = {
      "augmentations": []
    }
    if return_augmentation_history:
      augmentation_history['inputs'] = [img, boxes, landmarks]

    if random.random() <= self.prob_crop:
      if self.debug:
        print('applying crop')
      img, boxes, landmarks = crop(img, boxes, landmarks, is_bbox_safe=self.crop_is_bbox_safe,
                                   min_box_target_size=self.crop_min_box_target_size)
      if return_augmentation_history:
        augmentation_history['augmentations'].append('crop')
        augmentation_history['crop'] = [img, boxes, landmarks]
      if self.debug:
        print('boxes after crop:', boxes)

    # pre downscale
    if self.with_pre_downscale:
      if self.debug:
        print('applying pre downscale')
      pre_downscale_size = 1.5 * resize
      img, boxes, landmarks = resize_to_max(img, boxes, landmarks, pre_downscale_size)
      if return_augmentation_history:
        augmentation_history['augmentations'].append('pre_downscale')
        augmentation_history['pre_downscale'] = [img, boxes, landmarks]

    if random.random() < self.prob_rotate:
      if self.debug:
        print('applying rotation')
      square_size = max(img.shape[0], img.shape[1])
      aug_rot = self.albumentations_lib.Compose([
        # pad borders to avoid cutting out parts of image when rotating it
        self.albumentations_lib.augmentations.transforms.PadIfNeeded(p=1.0, min_height=int(square_size),
                                                                     min_width=int(square_size),
                                                                     border_mode=cv2.BORDER_CONSTANT),
        self.albumentations_lib.augmentations.transforms.Rotate(p=1.0, limit=(
          -self.max_rotation_angle, self.max_rotation_angle),
                                                                border_mode=cv2.BORDER_CONSTANT)
      ], keypoint_params=self.keypoint_params)

      res = aug_rot(image=img, keypoints=pack_abs_boxes_and_abs_landmarks(boxes, landmarks))
      img = res['image']
      boxes, landmarks = unpack_abs_boxes_and_abs_landmarks(res['keypoints'], len(boxes), len(landmarks))
      if return_augmentation_history:
        augmentation_history['augmentations'].append('rotate')
        augmentation_history['rotate'] = [img, boxes, landmarks]

    is_apply_anchor_based_sampling = random.random() <= self.prob_anchor_based_sampling and self.anchor_based_sampling_anchors is not None
    if is_apply_anchor_based_sampling:
      if self.debug:
        print('applying anchor_based_sampling')
      img, boxes, landmarks = anchor_based_sampling(img, boxes, landmarks, self.anchor_based_sampling_anchors,
                                                    max_scale=self.anchor_based_sampling_max_scale)
      if return_augmentation_history:
        augmentation_history['augmentations'].append('anchor_based_sampling')
        augmentation_history['anchor_based_sampling'] = [img, boxes, landmarks]

    if random.random() <= self.prob_flip:
      if self.debug:
        print('applying flip')
        print(boxes)
      res = self.albumentations_lib.Compose([
        self.albumentations_lib.augmentations.transforms.HorizontalFlip(p=1.0)
      ], keypoint_params=self.keypoint_params)(image=img, keypoints=pack_abs_boxes_and_abs_landmarks(boxes, landmarks))
      img = res['image']
      boxes, landmarks = unpack_abs_boxes_and_abs_landmarks(res['keypoints'], len(boxes), len(landmarks))
      if return_augmentation_history:
        augmentation_history['augmentations'].append('flip')
        augmentation_history['flip'] = [img, boxes, landmarks]

    if random.random() <= self.prob_stretch:
      stretch_x, stretch_y = self._get_stretch_shape(resize)
      img, boxes, landmarks = resize_by_ratio(img, boxes, landmarks, stretch_x / resize, stretch_y / resize)
      if return_augmentation_history:
        augmentation_history['augmentations'].append('stretch')
        augmentation_history['stretch'] = [img, boxes, landmarks]

    # crop to max output size and pad
    if not is_apply_anchor_based_sampling and self.resize_mode == 'crop_or_resize_to_fixed_and_random_pad' and random.random() < 0.5:
      if self.debug:
        print('applying resize_to_fixed_and_random_pad')
      img, boxes, landmarks = self._resize_to_fixed_and_pad(img, boxes, landmarks, resize, is_random_pad=True)
      if return_augmentation_history:
        augmentation_history['augmentations'].append('crop_and_random_pad_to_square')
        augmentation_history['crop_and_random_pad_to_square'] = [img, boxes, landmarks]
    else:
      if self.debug:
        print('applying crop_and_random_pad_to_square')
      img, boxes, landmarks = self._crop_to_max_and_pad(img, boxes, landmarks, resize, is_random_pad=True)
      if return_augmentation_history:
        augmentation_history['augmentations'].append('crop_and_random_pad_to_square')
        augmentation_history['crop_and_random_pad_to_square'] = [img, boxes, landmarks]

    if self.debug:
      print('applying color distortion')

    transformations = [
      self.albumentations_lib.augmentations.transforms.ToGray(p=self.prob_gray),
      self.albumentations_lib.augmentations.transforms.RandomGamma(p=self.prob_gamma, gamma_limit=self.gamma_limit),
      self.albumentations_lib.augmentations.transforms.HueSaturationValue(p=self.prob_hsv,
                                                                          hue_shift_limit=self.hue_shift_limit,
                                                                          sat_shift_limit=self.sat_shift_limit,
                                                                          val_shift_limit=self.val_shift_limit),
      self.albumentations_lib.augmentations.transforms.RGBShift(p=self.prob_rgb, r_shift_limit=self.r_shift_limit,
                                                                g_shift_limit=self.g_shift_limit,
                                                                b_shift_limit=self.b_shift_limit),
      self.albumentations_lib.augmentations.transforms.RandomBrightnessContrast(p=self.prob_brightness_contrast,
                                                                                brightness_limit=self.brightness_limit,
                                                                                contrast_limit=self.contrast_limit),
      self.albumentations_lib.augmentations.transforms.Blur(p=self.prob_blur,
                                                            blur_limit=int((self.blur_multiplier * resize) / 100)),
      self.albumentations_lib.augmentations.transforms.CoarseDropout(p=self.prob_dropout, max_holes=self.max_holes,
                                                                     max_height=int(self.max_hole_rel_size * resize),
                                                                     max_width=int(self.max_hole_rel_size * resize),
                                                                     fill_value=random.randint(0, 255))
    ]
    img = self.albumentations_lib.Compose(transformations)(image=img)['image']

    if return_augmentation_history:
      augmentation_history['augmentations'].append('color_distortion')
      augmentation_history['color_distortion'] = [img, boxes, landmarks]
      return augmentation_history

    return img, boxes, landmarks
