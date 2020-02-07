from colabsnippets.utils import rel_bbox_coords, min_bbox
from .pad_to_square import pad_to_square
from .utils import filter_abs_landmarks_out_of_borders, filter_abs_boxes_out_of_borders


def crop_and_random_pad_to_square(img, boxes, landmarks, image_size):
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
  img, boxes, landmarks = pad_to_square(img, boxes, landmarks, image_size, mode='random')
  return img, boxes, landmarks
