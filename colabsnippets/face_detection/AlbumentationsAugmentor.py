import random
import cv2

from ..utils import fix_boxes

class AlbumentationsAugmentor:
  def __init__(self, albumentations_lib, augment_lib):
    self.albumentations_lib = albumentations_lib
    # TODO: remove this
    self.augment_lib = augment_lib

    self.max_rotation_angle = 30
    self.max_stretch_x = 1.4
    self.max_stretch_y = 1.4
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

    self.prob_rotate = 0.5
    self.prob_flip = 0.5
    self.prob_stretch = 0.25
    self.prob_gamma = 0.5
    self.prob_hsv = 0.5
    self.prob_rgb = 0.5
    self.prob_brightness_contrast = 0.5
    self.prob_gray = 0.2
    self.prob_blur = 0.25
    self.prob_dropout = 0.25


  # assuming image is square with (size, size)
  def _get_stretch_shape(self, size):
    is_stretch_x = random.choice([True, False])

    stretch_factor = self.max_stretch_x if is_stretch_x else self.max_stretch_y
    stretched_dim = random.uniform(1.0, stretch_factor) * size
    scale_f = size / stretched_dim
    down_scaled_dim = int(round(scale_f * size))

    return (size, down_scaled_dim) if is_stretch_x else (down_scaled_dim, size)

  def augment(self, img, boxes = [], random_crop = None, resize = None):
    def abs_coords(bbox, img):
      height, width = img.shape[:2]
      min_x, min_y, max_x_or_w, max_y_or_h = bbox
      return [int(min_x * width), int(min_y * height), int(max_x_or_w * width), int(max_y_or_h * height)]

    def rel_coords(bbox, img):
      height, width = img.shape[:2]
      min_x, min_y, max_x_or_w, max_y_or_h = bbox
      return [min_x / width, min_y / height, max_x_or_w / width, max_y_or_h / height]

    fixed_boxes = []
    for  x, y, w, h in boxes:
      x, y = max(0, x), max(0, y)
      w, h = min(1.0 - x, w), min(1.0 - y, h)
      fixed_boxes.append((x, y, w, h))
    boxes = fixed_boxes


    bbox_params = self.albumentations_lib.BboxParams(format = 'coco', label_fields = ['labels'], min_area = 0.0, min_visibility = 0.0)

    transforms = self.albumentations_lib.augmentations.transforms
    Compose = self.albumentations_lib.Compose

    aug_rot = Compose([
      # pre downscale
      transforms.LongestMaxSize(p = 1.0, max_size = 1.5 * resize),
      transforms.PadIfNeeded(p = 1.0, min_height = int(1.5 * resize), min_width = int(1.5 * resize), border_mode = cv2.BORDER_CONSTANT),
      transforms.Rotate(p = self.prob_rotate, limit = (-self.max_rotation_angle, self.max_rotation_angle), border_mode = cv2.BORDER_CONSTANT)
    ], bbox_params)

    abs_boxes = fix_boxes([abs_coords(box, img) for box in boxes], max(img.shape[0:2]), 1)
    res = aug_rot(image = img, bboxes = abs_boxes, labels = ['' for box in boxes])

    res = Compose([
      transforms.RandomSizedBBoxSafeCrop(res['image'].shape[0], res['image'].shape[1], p = 1.0),
      transforms.LongestMaxSize(p = 1.0, max_size = resize)
    ], bbox_params)(**res)

    stretch_x, stretch_y = self._get_stretch_shape(resize)
    res = Compose([
      transforms.HorizontalFlip(p = self.prob_flip),
      transforms.Resize(stretch_y, stretch_x, p = self.prob_stretch),
      transforms.PadIfNeeded(p = 1.0, min_height = resize, min_width = resize, border_mode = cv2.BORDER_CONSTANT)
    ], bbox_params)(**res)

    img, boxes = res['image'], res['bboxes']

    # TODO: shear, rescale blur?
    transformations = [
      transforms.ToGray(p = self.prob_gray),
      transforms.RandomGamma(p = self.prob_gamma, gamma_limit = self.gamma_limit),
      transforms.HueSaturationValue(p = self.prob_hsv, hue_shift_limit = self.hue_shift_limit, sat_shift_limit = self.sat_shift_limit, val_shift_limit = self.val_shift_limit),
      transforms.RGBShift(p = self.prob_rgb, r_shift_limit = self.r_shift_limit, g_shift_limit = self.g_shift_limit, b_shift_limit = self.b_shift_limit),
      transforms.RandomBrightnessContrast(p = self.prob_brightness_contrast, brightness_limit = self.brightness_limit, contrast_limit = self.contrast_limit),
      transforms.Blur(p = self.prob_blur, blur_limit = int((self.blur_multiplier * resize) / 100)),
      transforms.CoarseDropout(p = self.prob_dropout, max_holes = self.max_holes, max_height = int(self.max_hole_rel_size * resize), max_width = int(self.max_hole_rel_size * resize), fill_value = random.randint(0, 255))
    ]
    img = Compose(transformations)(image = img)['image']
    # CUTOUT
    #img = transforms.Cutout(p = 1.0, num_holes=8, max_h_size=8, max_w_size=8).apply(img)

    # DISTORTIONS
    # TODO: a bit slow
    #img = transforms.OpticalDistortion(p = 1.0, distort_limit=0.05, shift_limit=0.05).apply(img)
    # TODO GridDistortion?
    #img = transforms.GridDistortion(p = 1.0, num_steps=5, distort_limit=0.3).apply(img)
    # TODO: very slow
    #img = transforms.ElasticTransform(p = 1.0, alpha=1, sigma=50, alpha_affine=50).apply(img)
    # TODO: module 'albumentations.augmentations.transforms' has no attribute 'RandomGridShuffle'
    #img = transforms.RandomGridShuffle(p = 1.0, grid=(3, 3)).apply(img)

    boxes = [rel_coords(box, res['image']) for box in boxes]
    return img, boxes

  def resize_and_to_square(self, image, boxes = None, image_size = None):
    # TODO: implement with albumentations
    return self.augment_lib.augment(image, boxes = boxes, pad_to_square = True, resize = image_size)