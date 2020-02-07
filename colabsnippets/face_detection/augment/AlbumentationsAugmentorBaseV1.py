import cv2

from colabsnippets.face_detection.augment.random_pad_to_square import random_pad_to_square
from colabsnippets.utils import fix_boxes, abs_bbox_coords, rel_bbox_coords, min_bbox


class AlbumentationsAugmentorBaseV1:
  def __init__(self, albumentations_lib):
    self.albumentations_lib = albumentations_lib
    self.bbox_params = self.albumentations_lib.BboxParams(format='coco', label_fields=['labels'], min_area=0.0,
                                                          min_visibility=0.0)
    self.ignore_log_augmentation_exception = False
    self.resize_mode = 'resize_to_max_and_center_pad'

  def _fix_rel_boxes(self, boxes):
    fixed_boxes = []
    for x, y, w, h in boxes:
      x0, y0, x1, y1 = x, y, x + w, y + h
      x0, y0 = max(0, x0), max(0, y0)
      x1, y1 = min(1.0, x1), min(1.0, y1)
      w, h = x1 - x0, y1 - y0
      if (x1 >= 1.0 or y1 >= 1.0) or w <= 0 or h <= 0:
        continue
      fixed_boxes.append((x0, y0, w, h))
    return fixed_boxes

  def _fix_abs_boxes(self, abs_boxes, hw):
    return [abs_bbox_coords(rel_box, hw) for rel_box in
            self._fix_rel_boxes([rel_bbox_coords(abs_box, hw) for abs_box in abs_boxes])]

  def _crop_and_random_pad_to_square(self, img, boxes, image_size):
    im_h, im_w = img.shape[0:2]
    rx, ry, rw, rh = min_bbox(rel_bbox_coords(abs_box, [im_h, im_w]) for abs_box in boxes)
    rcx, rcy = [int((rx + (rw / 2)) * im_w), int((ry + (rh / 2)) * im_h)]
    crop_x0 = int(max(0, rcx - (image_size / 2)))
    crop_y0 = int(max(0, rcy - (image_size / 2)))
    crop_x1 = int(min(im_w, rcx + (image_size / 2)))
    crop_y1 = int(min(im_h, rcy + (image_size / 2)))
    img = img[crop_y0:crop_y1, crop_x0:crop_x1, :]
    boxes = self._fix_abs_boxes([(x - crop_x0, y - crop_y0, w, h) for x, y, w, h in boxes], img.shape[0:2])
    img, boxes = random_pad_to_square(img, boxes, image_size)
    return img, boxes

  def _resize_to_max_and_center_pad(self, img, boxes, image_size):
    res = self.albumentations_lib.Compose([
      self.albumentations_lib.augmentations.transforms.LongestMaxSize(p=1.0, max_size=image_size),
      self.albumentations_lib.augmentations.transforms.PadIfNeeded(p=1.0, min_height=image_size, min_width=image_size,
                                                                   border_mode=cv2.BORDER_CONSTANT)
    ], self.bbox_params)(image=img, bboxes=boxes, labels=['' for _ in boxes])
    return res['image'], res['bboxes']

  def _augment_abs_boxes(self, img, boxes, image_size):
    raise Exception("AlbumentationsAugmentorBase - _augment_abs_boxes not implemented")

  def augment(self, img, boxes=[], image_size=None):
    try:
      _boxes = fix_boxes([abs_bbox_coords(box, img.shape[0:2]) for box in self._fix_rel_boxes(boxes)],
                         max(img.shape[0:2]), 1)
      _img, _boxes = self._augment_abs_boxes(img, _boxes, image_size)
      _boxes = [rel_bbox_coords(box, _img.shape[0:2]) for box in _boxes]
      return _img, _boxes
    except Exception as e:
      if not self.ignore_log_augmentation_exception:
        print("failed to augment")
        print(e)
      return self.resize_and_to_square(img, boxes=boxes, image_size=image_size)

  def resize_and_to_square(self, img, boxes=[], image_size=None):
    boxes = fix_boxes([abs_bbox_coords(box, img.shape[0:2]) for box in self._fix_rel_boxes(boxes)], max(img.shape[0:2]),
                      1)
    if self.resize_mode == 'resize_to_max_and_center_pad':
      img, boxes = self._resize_to_max_and_center_pad(img, boxes, image_size)
    elif self.resize_mode == 'crop_and_random_pad_to_square':
      img, boxes = self._crop_and_random_pad_to_square(img, boxes, image_size)
    else:
      raise Exception('unkown resize_mode ' + str(self.resize_mode))
    boxes = [rel_bbox_coords(box, img.shape[0:2]) for box in boxes]
    return img, boxes
