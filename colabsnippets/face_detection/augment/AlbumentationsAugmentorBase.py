from colabsnippets.utils import fix_boxes, abs_bbox_coords, rel_bbox_coords
from .crop_and_random_pad_to_square import crop_and_random_pad_to_square
from .pad_to_square import pad_to_square
from .resize_to_max import resize_to_max


def abs_landmarks_coords(landmarks, hw):
  height, width = hw[:2]
  return [(x * width, y * height) for x, y in landmarks]


def rel_landmarks_coords(landmarks, hw):
  height, width = hw[:2]
  return [(x / width, y / height) for x, y in landmarks]


class AlbumentationsAugmentorBase:
  def __init__(self, albumentations_lib):
    self.albumentations_lib = albumentations_lib
    self.bbox_params = self.albumentations_lib.BboxParams(format='coco', label_fields=['labels'], min_area=0.0,
                                                          min_visibility=0.0)
    self.keypoint_params = self.albumentations_lib.KeypointParams(format='xy', remove_invisible=False)
    self.ignore_log_augmentation_exception = False
    self.fallback_on_augmentation_exception = True
    self.resize_mode = 'resize_to_max_and_center_pad'

  def _resize_to_max_and_center_pad(self, img, boxes, landmarks, image_size):
    img, boxes, landmarks = resize_to_max(img, boxes, landmarks, image_size)
    img, boxes, landmarks = pad_to_square(img, boxes, landmarks, image_size, mode='center')
    return img, boxes, landmarks

  def _augment_abs_boxes(self, img, abs_boxes, abs_landmark, image_size):
    raise Exception("AlbumentationsAugmentorBase - _augment_abs_boxes not implemented")

  def augment(self, img, boxes=[], landmarks=[], image_size=None):
    try:
      _boxes = fix_boxes([abs_bbox_coords(box, img.shape[0:2]) for box in boxes],
                         max(img.shape[0:2]), 1)
      _landmarks = [abs_landmarks_coords(l, img.shape[0:2]) for l in landmarks]
      _img, _boxes, _landmarks = self._augment_abs_boxes(img, _boxes, _landmarks, image_size)
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
    if self.resize_mode == 'resize_to_max_and_center_pad':
      img, boxes, landmarks = self._resize_to_max_and_center_pad(img, boxes, landmarks, image_size)
    elif self.resize_mode == 'crop_and_random_pad_to_square':
      img, boxes, landmarks = crop_and_random_pad_to_square(img, boxes, landmarks, image_size)
    else:
      raise Exception('unkown resize_mode ' + str(self.resize_mode))
    boxes = [rel_bbox_coords(box, img.shape[0:2]) for box in boxes]
    landmarks = [rel_landmarks_coords(l, img.shape[0:2]) for l in landmarks]
    return img, boxes, landmarks
