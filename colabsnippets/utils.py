import json
import os
import random

import tensorflow as tf
from numpy import Inf

from .drive import get_global_drive_instance


def load_json(json_file_path):
  with open(json_file_path) as json_file:
    return json.load(json_file)


def shuffle_array(arr):
  arr_clone = arr[:]
  random.shuffle(arr_clone)
  return arr_clone


def load_json_if_exists(filepath):
  return load_json(filepath) if os.path.exists(filepath) else []


# auto recompile ops in case of new batch size
def forward_factory(compile_forward_op, batch_size, image_size):
  X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
  forward_op = compile_forward_op(X)

  def forward(sess, batch_x):
    local_X, local_forward_op = X, forward_op
    if batch_x.shape[0] != X.shape[0]:
      local_X = tf.placeholder(tf.float32, [batch_x.shape[0], image_size, image_size, 3])
      local_forward_op = compile_forward_op(local_X)
    return sess.run(local_forward_op, feed_dict={local_X: batch_x})

  return forward


def mk_dir_if_not_exists(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def flatten_list(l):
  return [item for sublist in l for item in sublist]


def try_upload_file(filename, drive_upload_folder_id):
  try:
    upload = get_global_drive_instance().CreateFile(
      {"title": filename, "parents": [{"kind": "drive#fileLink", "id": drive_upload_folder_id}]})
    upload.SetContentFile(filename)
    upload.Upload()
  except Exception as e:
    print("failed to upload " + filename)
    print(e)


def gpu_session(callback, device_name='/gpu:0'):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = True
  with tf.Session(config=config) as session:
    with tf.device(device_name):
      return callback(session)


def min_bbox_from_pts(pts):
  min_x, min_y, max_x, max_y = 1.0, 1.0, 0, 0
  for x, y in pts:
    min_x = x if x < min_x else min_x
    min_y = y if y < min_y else min_y
    max_x = max_x if x < max_x else x
    max_y = max_y if y < max_y else y

  return [min_x, min_y, max_x - min_x, max_y - min_y]


def min_bbox(boxes):
  pts = []
  for box in boxes:
    x, y, w, h = box
    pts.append([x, y])
    pts.append([x + w, y + h])
  return min_bbox_from_pts(pts)


def num_in_range(val, min_val, max_val):
  return min(max(min_val, val), max_val)


def abs_bbox_coords(bbox, hw):
  height, width = hw[:2]
  min_x, min_y, max_x_or_w, max_y_or_h = bbox
  return [int(min_x * width), int(min_y * height), int(max_x_or_w * width), int(max_y_or_h * height)]


def rel_bbox_coords(bbox, hw):
  height, width = hw[:2]
  min_x, min_y, max_x_or_w, max_y_or_h = bbox
  return [min_x / width, min_y / height, max_x_or_w / width, max_y_or_h / height]


def filter_abs_boxes(abs_boxes, min_box_size_px=0, max_box_size_px=Inf):
  filtered_boxes = []
  for box in abs_boxes:
    _, __, w, h = box
    if min(w, h) < min_box_size_px or min(w, h) > max_box_size_px:
      continue
    filtered_boxes.append(box)
  return filtered_boxes


def fix_boxes(boxes, image_size, min_box_size_px):
  out_boxes = []
  for box in boxes:
    x, y, w, h = box

    if (image_size * w) <= min_box_size_px or (image_size * h) <= min_box_size_px:
      continue

    out_boxes.append(box)

  return out_boxes
