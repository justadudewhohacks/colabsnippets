import random
import time
import tensorflow as tf
import numpy as np

from ...utils import load_json, try_upload_file, save_weights, gpu_session, fix_boxes
from ..AlbumentationsAugmentor import AlbumentationsAugmentor
from .DataLoader import DataLoader
from .EpochStatsFPN import EpochStatsFPN

def get_net_vars(net_name):
  all_vars = tf.global_variables(net_name)
  net_vars = []
  for var in all_vars:
    if not '/Adam' in var.name:
      net_vars.append(var)
  return net_vars

class TrainFPN():
  def __init__(self, args, num_reduction_ops = 4):
    self.net = args['net']
    self.model_name = args['model_name']
    self.start_epoch = args['start_epoch']
    self.learning_rate = args['learning_rate']
    self.batch_size = args['batch_size']
    self.augmentation_prob = args['augmentation_prob']
    self.no_object_scale = args['no_object_scale']
    self.object_scale = args['object_scale']
    self.coord_scale = args['coord_scale']
    self.drive_upload_epoch_txts_folder_id = args['drive_upload_epoch_txts_folder_id']
    self.drive_upload_checkpoints_folder_id = args['drive_upload_checkpoints_folder_id']
    self.image_sizes = args["image_sizes"]
    albumentations_lib = args["albumentations_lib"]
    augment_lib = args["augment_lib"]
    min_box_size_px = args['min_box_size_px']

    self.num_reduction_ops = num_reduction_ops

    ibug_challenge_data = load_json('./ibug_challenge_data.json')
    face_detection_scrapeddb_data = load_json('./face_detection_scrapeddb_data.json')
    wider_trainData = load_json('./wider_trainData.json')
    train_data = wider_trainData + ibug_challenge_data + face_detection_scrapeddb_data
    self.data_loader = DataLoader(train_data, start_epoch = self.start_epoch, image_augmentor = AlbumentationsAugmentor(albumentations_lib, augment_lib), augmentation_prob = self.augmentation_prob, min_box_size_px = min_box_size_px)

    print('starting at epoch:', self.start_epoch)
    print('---------------------------')
    print('learning_rate:', self.learning_rate)
    print('batch_size:', self.batch_size)
    print('augmentation_prob:', self.augmentation_prob)
    print('num_reduction_ops:', self.num_reduction_ops)
    print('no_object_scale:', self.no_object_scale)
    print('train samples:', len(train_data))
    print('image sizes:', self.image_sizes)
    print('---------------------------')

  def train(self):
    if (self.start_epoch != 0):
      self.net.load_weights("{}_epoch{}".format(self.model_name, self.start_epoch - 1), net_json_file = self.net.name + '.json')
      print('done loading weights')
    else:
      self.net.init_trainable_weights()

    log_file = open('./log.txt', 'w')
    def run(sess):
      # compile graphs
      print('compiling graphs')
      forward_train_ops_by_image_size = {}
      for image_size in self.image_sizes:
        out_num_cells = int(image_size / (2**self.num_reduction_ops))
        forward_train_ops_by_image_size[image_size] = self.net.forward_factory(
            sess, self.batch_size, image_size, out_num_cells = out_num_cells, with_train_ops = True,
            learning_rate = self.learning_rate, object_scale = self.object_scale, coord_scale = self.coord_scale, no_object_scale = self.no_object_scale, apply_scale_loss = tf.abs)

      sess.run(tf.global_variables_initializer())

      print('start training')
      epoch_stats = EpochStatsFPN()
      # TODO: anchor epochs
      data_loader = self.data_loader
      while True:
        current_epoch = data_loader.epoch
        image_size = random.choice(self.image_sizes)
        forward_train = forward_train_ops_by_image_size[image_size]

        ts = time.time()
        batch_x, batch_gt_boxes = data_loader.next_batch(self.batch_size, image_size)
        preds = forward_train(batch_x, batch_gt_boxes)

        stats = epoch_stats.update(preds, batch_gt_boxes)
        format_array = lambda arr: str(["{:.4f}".format(l) for l in arr])
        add_loss_entry = lambda entry: ", {}= {} ({})".format(entry, format_array(preds[entry]), format_array(preds[entry] / np.array(stats["num_gt_anchors_by_stage"])))
        log_file.write(
            "epoch " + str(current_epoch) + ", (" + str(data_loader.current_idx) + " of " + str(data_loader.get_end_idx()) + ")"
              + ", image_size= {}".format(image_size)
              + ", num_gt_boxes= {}".format(stats["num_gt_boxes"])
              + ", loss= {:.4f} ({:.4f})".format(preds["loss"], preds["loss"] / stats["num_gt_boxes"])
              + add_loss_entry("weighted_object_losses_by_stage")
              + add_loss_entry("weighted_no_object_losses_by_stage")
              + add_loss_entry("weighted_offset_losses_by_stage")
              + add_loss_entry("weighted_scales_losses_by_stage")
              + ", time= " + str((time.time() - ts) * 1000) + "ms \n")

        if current_epoch != data_loader.epoch:
          print('next epoch: ' + str(data_loader.epoch))
          checkpoint_name = "{}_epoch{}".format(self.model_name, current_epoch)
          save_weights(get_net_vars(self.net.name), checkpoint_name)

          epoch_txt_filename = "{}_epoch{}.txt".format(self.model_name, current_epoch)
          epoch_txt = open(epoch_txt_filename, 'w')
          epoch_txt.write('learning_rate= ' + str(self.learning_rate) + '\n')
          epoch_txt.write('object_scale= ' + str(self.object_scale) + '\n')
          epoch_txt.write('no_object_scale= ' + str(self.no_object_scale) + '\n')
          epoch_txt.write('coord_scale= ' + str(self.coord_scale) + '\n')
          epoch_txt.write('batch_size= ' + str(self.batch_size) + '\n')
          epoch_txt.write('augmentation_prob= ' + str(self.augmentation_prob) + '\n')
          epoch_stats.write_stats(epoch_txt)
          epoch_txt.close()

          try_upload_file(epoch_txt_filename, self.drive_upload_epoch_txts_folder_id)
          try_upload_file(checkpoint_name + ".npy", self.drive_upload_checkpoints_folder_id)

          epoch_stats = EpochStatsFPN()

    gpu_session(run)