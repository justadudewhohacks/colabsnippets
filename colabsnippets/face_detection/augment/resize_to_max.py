from .resize_to_fixed import resize_to_fixed


def resize_to_max(img, boxes, landmarks, image_size, is_relative_coords=False):
  im_h, im_w = img.shape[0:2]
  if max(im_h, im_w) > image_size:
    return resize_to_fixed(img, boxes, landmarks, image_size, is_relative_coords=is_relative_coords)
  return img, boxes, landmarks
