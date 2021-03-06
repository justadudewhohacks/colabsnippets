import types

from ...BatchLoader import BatchLoader
from ...utils import load_json, load_json_if_exists, fix_boxes, min_bbox_from_pts


def json_boxes_to_array(boxes):
  out_boxes = []
  for box in boxes:
    x, y, w, h = box['x'], box['y'], box['width'], box['height']
    out_box = (x, y, w, h)
    if w <= 0 or h <= 0:
      raise Exception("box has invalid width or height: {}".format(out_box))
    for val in out_box:
      if val < -0.5 or val > 1.5:
        raise Exception("box is probably not a valid relative box: {}".format(out_box))
    out_boxes.append(out_box)
  return out_boxes


def resolve_image_path(data):
  db = data['db']
  img_file = data['file']
  img_dir = "images-shard{}".format(data['shard']) if 'shard' in data else 'images'
  img_path = "./data/{}/{}/{}".format(db, img_dir, img_file)
  return img_path


class DataLoader(BatchLoader):
  def __init__(self, data_or_data_getter, image_augmentor=None, inputs_to_square=True, start_epoch=None, is_test=False,
               min_box_size_px=8,
               augmentation_prob=0.0, extract_data_labels_fallback=None):
    self.image_augmentor = image_augmentor
    self.augmentation_prob = augmentation_prob
    self.min_box_size_px = min_box_size_px
    self.inputs_to_square = inputs_to_square

    celeba_landmarks_by_file = load_json_if_exists('./data/celeba/landmarks.json')
    face_detection_scrapeddb_boxes_by_file = load_json_if_exists('./data/face_detection_scrapeddb/boxes.json')
    helen_boxes_by_file = load_json_if_exists('./data/helen/boxes.json')
    ibug_boxes_by_file = load_json_if_exists('./data/ibug/boxes.json')
    lfpw_boxes_by_file = load_json_if_exists('./data/lfpw/boxes.json')
    afw_boxes_by_file = load_json_if_exists('./data/afw/boxes.json')
    thw_boxes_by_file = load_json_if_exists('./data/300w/boxes.json')
    mafa_train_boxes_by_file = load_json_if_exists('./data/MAFA_train/boxes.json')
    mafa_test_boxes_by_file = load_json_if_exists('./data/MAFA_test/boxes.json')
    ufdd_val_boxes_by_file = load_json_if_exists('./data/UFDD_val/boxes.json')
    face_detection_scrapeddb_landmarks_by_file = load_json_if_exists('./data/face_detection_scrapeddb/landmarks.json')
    ibug_all_landmarks_by_file = load_json_if_exists('./data/ibug/landmarks.json')
    mafa_train_landmarks_by_file = load_json_if_exists('./data/MAFA_train/landmarks.json')
    ufdd_val_landmarks_by_file = load_json_if_exists('./data/UFDD_val/landmarks.json')

    def extract_data_labels(data):
      db = data['db']
      img_file = data['file']
      if db == 'celeba':
        landmarks = celeba_landmarks_by_file[img_file]
        x, y, w, h = min_bbox_from_pts(landmarks)
        padding = 1.5

        x = x - (0.5 * padding * w)
        y = y - (0.5 * padding * h)
        w = w + (padding * w)
        h = h + (padding * h)

        return [(x, y, w, h)]
      if db == 'WIDER' or db == 'FDDB' or db == 'celeba_gen_32_160_32' or db == 'celeba_gen_64_320_32':
        boxes_file = img_file.replace('.jpg', '.json')
        boxes_dir = "boxes-shard{}".format(data['shard']) if 'shard' in data else 'boxes'
        boxes_path = "./data/{}/{}/{}".format(db, boxes_dir, boxes_file)
        boxes = load_json(boxes_path)
        if db == 'WIDER' or db == 'FDDB':
          boxes = json_boxes_to_array(boxes)
        landmarks = []
        if db == 'WIDER':
          landmarks_file = img_file.replace('.jpg', '.json')
          landmarks_dir = "landmarks-shard{}".format(data['shard']) if 'shard' in data else 'landmarks'
          landmarks_path = "./data/{}/{}/{}".format(db, landmarks_dir, landmarks_file)
          landmarks = load_json_if_exists(landmarks_path)
          landmarks = landmarks if landmarks is not None else []
        return boxes, landmarks
      if db == 'face_detection_scrapeddb':
        return face_detection_scrapeddb_boxes_by_file[img_file], face_detection_scrapeddb_landmarks_by_file[img_file]
      if db == 'helen':
        return helen_boxes_by_file[img_file], ibug_all_landmarks_by_file[img_file]
      if db == 'ibug':
        return ibug_boxes_by_file[img_file], ibug_all_landmarks_by_file[img_file]
      if db == 'afw':
        return afw_boxes_by_file[img_file], ibug_all_landmarks_by_file[img_file]
      if db == 'lfpw':
        return lfpw_boxes_by_file[img_file], ibug_all_landmarks_by_file[img_file]
      if db == '300w':
        return thw_boxes_by_file[img_file], ibug_all_landmarks_by_file[img_file]
      if db == 'MAFA_train':
        return mafa_train_boxes_by_file[img_file], mafa_train_landmarks_by_file[img_file]
      if db == 'MAFA_test':
        return mafa_test_boxes_by_file[img_file], []
      if db == 'UFDD_val':
        return ufdd_val_boxes_by_file[img_file], ufdd_val_landmarks_by_file[img_file]
      if db == 'celeba_face_clusters':
        boxes_file = img_file.replace('.jpg', '.json')
        boxes_dir = 'generated-boxes'
        boxes_path = "./data/{}/{}/{}".format(db, boxes_dir, boxes_file)
        boxes = load_json(boxes_path)
        return boxes, []
      if extract_data_labels_fallback is not None:
        return extract_data_labels_fallback(data)
      raise Exception("extract_data_labels - unknown db '{}'".format(db))

    BatchLoader.__init__(
      self,
      data_or_data_getter if type(data_or_data_getter) is types.FunctionType else lambda: data_or_data_getter,
      resolve_image_path,
      extract_data_labels,
      start_epoch=start_epoch,
      is_test=is_test
    )

  def load_image_and_labels_batch(self, datas, image_size):
    batch_x, batch_y = [], []
    for data in datas:
      image = self.load_image(data)
      boxes, landmarks = self.extract_data_labels(data)
      if self.inputs_to_square:
        image, boxes, landmarks = self.image_augmentor.make_inputs(image, boxes=boxes, landmarks=landmarks,
                                                                   image_size=image_size,
                                                                   augmentation_prob=self.augmentation_prob)
      else:
        pass
      batch_x.append(image)
      batch_y.append([fix_boxes(boxes, image_size, self.min_box_size_px), landmarks])

    return batch_x, batch_y

  def next_batch(self, batch_size, image_size=None):
    batch_x, batch_y = [], []
    while len(batch_x) < batch_size:
      next_batch = BatchLoader.next_batch(self, 1, image_size=image_size)
      if next_batch is None:
        return None
      image, labels = next_batch
      image, labels = image[0], labels[0]
      boxes, landmarks = labels
      boxes = fix_boxes(boxes, image_size, self.min_box_size_px)
      if len(boxes) > 0:
        batch_x.append(image)
        batch_y.append([boxes, landmarks])
    return batch_x, batch_y
