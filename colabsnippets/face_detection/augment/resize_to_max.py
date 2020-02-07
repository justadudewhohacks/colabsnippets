from .resize_by_ratio import resize_by_ratio


def resize_to_max(img, boxes, landmarks, image_size):
  im_h, im_w = img.shape[0:2]
  if max(im_h, im_w) > image_size:
    s = image_size / max(im_h, im_w)
    img, boxes, landmarks = resize_by_ratio(img, boxes, landmarks, s, s)
  return img, boxes, landmarks
