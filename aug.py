import albumentations
import cv2

from colabsnippets.face_detection.augment import AlbumentationsAugmentor
from colabsnippets.face_detection.augment.anchor_based_sampling import anchor_based_sampling
from colabsnippets.face_detection.augment.crop import crop
from colabsnippets.face_detection.augment.resize_to_max import resize_to_max
from colabsnippets.face_detection.fpn.DataLoader import json_boxes_to_array
from colabsnippets.utils import load_json, abs_bbox_coords, filter_abs_boxes, rel_bbox_coords

img = cv2.imread('./0_Parade_marchingband_1_8.jpg')
boxes = json_boxes_to_array(load_json('./0_Parade_marchingband_1_8.json'))
# img = cv2.imread('./single_face.jpeg')
# boxes = json_boxes_to_array(load_json('./single_face.json'))

augmentor = AlbumentationsAugmentor(albumentations_lib=albumentations)
# augmentor.debug = True
augmentor.resize_mode = 'crop_and_random_pad_to_square'
augmentor.return_augmentation_history = True
augmentor.fallback_on_augmentation_exception = False

min_box_px_size = 1
augmentor.prob_rotate = 1.0


def run2():
  while True:
    abs_boxes = filter_abs_boxes([abs_bbox_coords(rel_box, img.shape[0:2]) for rel_box in boxes], min_box_px_size)
    rel_boxes = [rel_bbox_coords(abs_box, img.shape[0:2]) for abs_box in abs_boxes]
    rel_landmarks = []
    outputs = augmentor.augment(img, rel_boxes, rel_landmarks, 640)

    print(outputs['augmentations'])

    keys = ['inputs', 'crop', 'pre_downscale', 'rotate', 'anchor_based_sampling', 'flip', 'stretch',
            'crop_and_random_pad_to_square', 'color_distortion']

    for aug in keys:
      if aug in outputs:
        out, out_boxes, out_landmarks = outputs[aug]
        out, out_boxes, out_landmarks = resize_to_max(out, out_boxes, out_landmarks, 640, is_relative_coords=True)

        out_boxes = [abs_bbox_coords(out_box, out.shape[0:2]) for out_box in out_boxes]
        # print(out_boxes)
        for x, y, w, h in out_boxes:
          cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # print(out.shape, len(out_boxes))
        if out.shape[0] != 640 or out.shape[1] != 640:
          print(out.shape)
        cv2.imshow(aug, out)
    cv2.waitKey()
    cv2.destroyAllWindows()


def run1():
  while True:
    abs_boxes = filter_abs_boxes([abs_bbox_coords(rel_box, img.shape[0:2]) for rel_box in boxes], min_box_px_size)
    out, out_boxes = img, abs_boxes
    out, out_boxes = crop(out, out_boxes, is_bbox_safe=False, min_box_target_size=0)
    out, out_boxes = anchor_based_sampling(out, out_boxes, [16, 32, 64, 128, 256], max_scale=3.0)
    # print(out_boxes)
    for x, y, w, h in out_boxes:
      cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # print(out.shape, len(out_boxes))
    cv2.imshow("foo", out)
    cv2.waitKey()


run2()
