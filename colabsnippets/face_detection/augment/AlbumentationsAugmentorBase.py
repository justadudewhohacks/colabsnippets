import random

from colabsnippets.utils import fix_boxes, abs_bbox_coords, rel_bbox_coords, rel_landmarks_coords, abs_landmarks_coords, \
  min_bbox
from .pad_to_square import pad_to_square
from .resize_to_fixed import resize_to_fixed
from .resize_to_max import resize_to_max
from .utils import filter_abs_landmarks_out_of_borders, filter_abs_boxes_out_of_borders


class AlbumentationsAugmentorBase:
  def __init__(self, albumentations_lib):
    self.albumentations_lib = albumentations_lib
    self.bbox_params = self.albumentations_lib.BboxParams(format='coco', label_fields=['labels'], min_area=0.0,
                                                          min_visibility=0.0)
    self.keypoint_params = self.albumentations_lib.KeypointParams(format='xy', remove_invisible=False)
    self.ignore_log_augmentation_exception = False
    self.fallback_on_augmentation_exception = True
    self.resize_mode = 'resize_to_max_and_center_pad'
    self.return_augmentation_history = False

  def _crop_to_max_and_pad(self, img, boxes, landmarks, image_size, is_random_pad=False):
    im_h, im_w = img.shape[0:2]
    rx, ry, rw, rh = min_bbox(rel_bbox_coords(abs_box, [im_h, im_w]) for abs_box in boxes)
    rcx, rcy = [int((rx + (rw / 2)) * im_w), int((ry + (rh / 2)) * im_h)]
    crop_x0 = int(max(0, rcx - (image_size / 2)))
    crop_y0 = int(max(0, rcy - (image_size / 2)))
    crop_x1 = int(min(im_w, rcx + (image_size / 2)))
    crop_y1 = int(min(im_h, rcy + (image_size / 2)))
    img = img[crop_y0:crop_y1, crop_x0:crop_x1, :]
    boxes = [[x - crop_x0, y - crop_y0, w, h] for x, y, w, h in boxes]
    landmarks = [[[x - crop_x0, y - crop_y0] for x, y in l] for l in landmarks]
    boxes = filter_abs_boxes_out_of_borders(boxes, img.shape[0:2])
    landmarks = filter_abs_landmarks_out_of_borders(landmarks, img.shape[0:2])
    img, boxes, landmarks = pad_to_square(img, boxes, landmarks, image_size,
                                          mode='random' if is_random_pad else 'center')
    return img, boxes, landmarks

  def _resize_to_max_and_center_pad(self, img, boxes, landmarks, image_size):
    img, boxes, landmarks = resize_to_max(img, boxes, landmarks, image_size)
    img, boxes, landmarks = pad_to_square(img, boxes, landmarks, image_size, mode='center')
    return img, boxes, landmarks

  def _resize_to_fixed_and_pad(self, img, boxes, landmarks, image_size, is_random_pad=False):
    img, boxes, landmarks = resize_to_fixed(img, boxes, landmarks, image_size)
    img, boxes, landmarks = pad_to_square(img, boxes, landmarks, image_size,
                                          mode='random' if is_random_pad else 'center')
    return img, boxes, landmarks

  def _augment_abs_boxes(self, img, abs_boxes, abs_landmark, image_size, return_augmentation_history=False):
    raise Exception("AlbumentationsAugmentorBase - _augment_abs_boxes not implemented")

  def augment(self, img, boxes=[], landmarks=[], image_size=None):
    try:
      _boxes = fix_boxes([abs_bbox_coords(box, img.shape[0:2]) for box in boxes],
                         max(img.shape[0:2]), 1)
      _landmarks = [abs_landmarks_coords(l, img.shape[0:2]) for l in landmarks]
      out = self._augment_abs_boxes(img, _boxes, _landmarks, image_size,
                                    return_augmentation_history=self.return_augmentation_history)
      if self.return_augmentation_history:
        for aug in (['inputs'] + out['augmentations']):
          _img, _boxes, _landmarks = out[aug]
          _boxes = [rel_bbox_coords(box, _img.shape[0:2]) for box in _boxes]
          _landmarks = [rel_landmarks_coords(l, _img.shape[0:2]) for l in _landmarks]
          out[aug] = [_img, _boxes, _landmarks]
        return out
      else:
        _img, _boxes, _landmarks = out
        _boxes = [rel_bbox_coords(box, _img.shape[0:2]) for box in _boxes]
        _landmarks = [rel_landmarks_coords(l, _img.shape[0:2]) for l in _landmarks]
        return _img, _boxes, _landmarks
    except Exception as e:
      if not self.fallback_on_augmentation_exception:
        raise e
      if not self.ignore_log_augmentation_exception:
        print("failed to augment")
        print(e)
      return self.resize_and_to_square(img, boxes=boxes, landmarks=landmarks, image_size=image_size)

  def resize_and_to_square(self, img, boxes=[], landmarks=[], image_size=None):
    boxes = fix_boxes([abs_bbox_coords(box, img.shape[0:2]) for box in boxes], max(img.shape[0:2]),
                      1)
    landmarks = [abs_landmarks_coords(l, img.shape[0:2]) for l in landmarks]

    resize_mode = self.resize_mode
    if resize_mode == 'crop_or_resize_to_fixed_and_random_pad':
      resize_mode = random.choice(['crop_and_random_pad_to_square', 'resize_to_fixed_and_random_pad'])

    if resize_mode == 'resize_to_max_and_center_pad':
      img, boxes, landmarks = self._resize_to_max_and_center_pad(img, boxes, landmarks, image_size)
    elif resize_mode == 'resize_to_fixed_and_center_pad':
      img, boxes, landmarks = self._resize_to_fixed_and_pad(img, boxes, landmarks, image_size, is_random_pad=False)
    elif resize_mode == 'resize_to_fixed_and_random_pad':
      img, boxes, landmarks = self._resize_to_fixed_and_pad(img, boxes, landmarks, image_size, is_random_pad=True)
    elif resize_mode == 'crop_and_random_pad_to_square':
      img, boxes, landmarks = self._crop_to_max_and_pad(img, boxes, landmarks, image_size, is_random_pad=True)
    else:
      raise Exception('unknown resize_mode ' + str(resize_mode))
    boxes = [rel_bbox_coords(box, img.shape[0:2]) for box in boxes]
    landmarks = [rel_landmarks_coords(l, img.shape[0:2]) for l in landmarks]
    return img, boxes, landmarks

  def make_inputs(self, image, boxes=[], landmarks=[], image_size=None, augmentation_prob=1.0):
    if random.random() < augmentation_prob:
      return self.augment(image, boxes=boxes, landmarks=landmarks,
                          image_size=image_size)
    return self.resize_and_to_square(image, boxes=boxes, landmarks=landmarks,
                                     image_size=image_size)
