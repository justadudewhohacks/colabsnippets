from .FlipCropGrayAugmentor import FlipCropGrayAugmentor


class FlipGrayAugmentor(FlipCropGrayAugmentor):
  def __init__(self, albumentations_lib):
    super().__init__(albumentations_lib)
    self.prob_crop = 0.0
