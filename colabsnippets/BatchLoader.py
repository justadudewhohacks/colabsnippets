import cv2
import numpy as np

from .utils import shuffle_array

class BatchLoader:
  def __init__(
    self,
    get_epoch_data,
    resolve_image_path,
    extract_data_labels,
    is_test = False,
    start_epoch = None
  ):
    if not is_test and start_epoch == None:
      raise Exception('DataLoader - start_epoch has to be defined in train mode')

    self.get_epoch_data = get_epoch_data
    self.resolve_image_path = resolve_image_path
    self.extract_data_labels = extract_data_labels
    self.is_test = is_test
    self.epoch = start_epoch
    self.buffered_data = shuffle_array(self.get_epoch_data()) if not is_test else self.get_epoch_data()
    self.current_idx = 0

  def get_end_idx(self):
    return len(self.buffered_data)

  def load_image(self, data, rgb = True):
    img_file_path = self.resolve_image_path(data)
    img = cv2.imread(img_file_path)
    if rgb:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
      raise Exception('failed to read image from path: ' + img_file_path)
    return img

  def preprocess_image(self, image, image_size):
    return cv2.resize(image, (image_size, image_size))

  def load_image_batch(self, datas, image_size):
    preprocessed_imgs = []
    for data in datas:
      preprocessed_imgs.append(self.preprocess_image(self.load_image(data), image_size))
    return np.stack(preprocessed_imgs, axis = 0)

  def load_labels(self, datas):
    labels = []
    for data in datas:
      labels.append(self.extract_data_labels(data))
    return labels

  def load_image_and_labels_batch(self, datas, image_size):
    batch_x = self.load_image_batch(datas, image_size)
    batch_y = self.load_labels(datas)
    return batch_x, batch_y

  def next_batch(self, batch_size, image_size = 112):
    if batch_size < 1:
      raise Exception('BatchLoader.next_batch - invalid batch_size: ' + str(batch_size))


    from_idx = self.current_idx
    to_idx = self.current_idx + batch_size

    # end of epoch
    if (to_idx > len(self.buffered_data)):
      if self.is_test:
        to_idx = len(self.buffered_data)
        if to_idx == self.current_idx:
          return None
      else:
        self.epoch += 1
        self.buffered_data = self.buffered_data[from_idx:] + shuffle_array(self.get_epoch_data())
        from_idx = 0
        to_idx = batch_size

    self.current_idx = to_idx

    next_data = self.buffered_data[from_idx:to_idx]

    batch_x, batch_y = self.load_image_and_labels_batch(next_data, image_size)
    return batch_x, batch_y