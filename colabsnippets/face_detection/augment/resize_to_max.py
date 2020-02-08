from colabsnippets.utils import abs_bbox_coords, rel_bbox_coords, rel_landmarks_coords, abs_landmarks_coords
from .resize_by_ratio import resize_by_ratio


def resize_to_max(img, boxes, landmarks, image_size, is_relative_coords=False):
  im_h, im_w = img.shape[0:2]
  if max(im_h, im_w) > image_size:
    s = image_size / max(im_h, im_w)
    if is_relative_coords:
      boxes = [abs_bbox_coords(box, img.shape[0:2]) for box in boxes]
      landmarks = [abs_landmarks_coords(l, img.shape[0:2]) for l in landmarks]
    img, boxes, landmarks = resize_by_ratio(img, boxes, landmarks, s, s)
    if is_relative_coords:
      boxes = [rel_bbox_coords(box, img.shape[0:2]) for box in boxes]
      landmarks = [rel_landmarks_coords(l, img.shape[0:2]) for l in landmarks]
  return img, boxes, landmarks
