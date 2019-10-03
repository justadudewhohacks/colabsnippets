import cv2

from colabsnippets.face_detection.AlbumentationsAugmentorBase import AlbumentationsAugmentorBase


class FlipCropGrayAugmentor(AlbumentationsAugmentorBase):
  def __init__(self, albumentations_lib):
    super().__init__(albumentations_lib)

    self.prob_crop = 0.5
    self.prob_flip = 0.5
    self.prob_gray = 0.2

  def _augment_abs_boxes(self, img, boxes, resize):
    transforms = self.albumentations_lib.augmentations.transforms
    Compose = self.albumentations_lib.Compose

    res = Compose([
      transforms.RandomSizedBBoxSafeCrop(img.shape[0], img.shape[1], p=1.0),
      transforms.LongestMaxSize(p=1.0, max_size=resize),
      transforms.HorizontalFlip(p=self.prob_flip),
      transforms.ToGray(p=self.prob_gray),
      transforms.PadIfNeeded(p=1.0, min_height=resize, min_width=resize, border_mode=cv2.BORDER_CONSTANT)
    ], self.bbox_params)(image=img, bboxes=boxes, labels=['' for _ in boxes])

    img, boxes = res['image'], res['bboxes']
    return img, boxes
