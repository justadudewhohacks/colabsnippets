import random

import cv2

from colabsnippets.face_detection.augment.crop import crop
from colabsnippets.utils import load_json, abs_bbox_coords

from colabsnippets.face_detection.fpn.DataLoader import json_boxes_to_array

img = cv2.imread('./0_Parade_marchingband_1_8.jpg')
boxes = json_boxes_to_array(load_json('./0_Parade_marchingband_1_8.json'))


while True:
  abs_boxes = [abs_bbox_coords(rel_box, img.shape[0:2]) for rel_box in boxes]
  out, out_boxes = crop(img, abs_boxes, crop_range=0, is_bbox_safe=True, min_box_target_size=32)

  for box in out_boxes:
    x, y, w, h = abs_bbox_coords(box, out.shape)
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)

  print(out.shape, len(out_boxes))
  cv2.imshow("foo", out)
  cv2.waitKey()
