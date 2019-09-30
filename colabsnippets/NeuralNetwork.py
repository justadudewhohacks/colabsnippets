import json

import tensorflow as tf
import numpy as np

from .WeightInitializer import WeightInitializer
from .utils import load_json


class NeuralNetwork:
  def __init__(self, initialize_weights, name=None):
    self.initialize_weights = initialize_weights
    self.name = name

  def init_trainable_weights(self, weight_initializer=tf.keras.initializers.glorot_normal(),
                             bias_initializer=tf.keras.initializers.Zeros()):
    def initialize_weights_factory(initializer):
      def initialize_weights(name, shape):
        return tf.get_variable(name, initializer=initializer(shape))

      return initialize_weights

    self.initialize_weights(
      WeightInitializer(initialize_weights_factory(weight_initializer), initialize_weights_factory(bias_initializer)))

  def get_net_vars_in_initialization_order(self):
    existing_vars = tf.global_variables()
    vars = []

    def append_variable(name, shape):
      var_name = tf.get_variable_scope().name + '/' + name + ':0'
      matches = [v for v in existing_vars if v.name == var_name]
      if len(matches) < 1:
        raise Exception("get_net_vars_in_initialization_order - failed to find variable for name: " + var_name)
      if len(matches) > 1:
        raise Exception("get_net_vars_in_initialization_order - found multiple variables for name: " + var_name)
      vars.append(matches[0])

    self.initialize_weights(WeightInitializer(append_variable, append_variable), reuse=True)
    return vars

  def save_meta_json(self, filename):
    var_list = self.get_net_vars_in_initialization_order()
    meta_data = []
    for var in var_list:
      meta_data.append({'shape': var.get_shape().as_list(), 'name': var.name})
    meta_json = open(filename + '.json', 'w')
    meta_json.write(json.dumps(meta_data))
    meta_json.close()

  def save_weights(self, filename):
    var_list = self.get_net_vars_in_initialization_order()
    checkpoint_data = np.array([], dtype='float32')
    for var in var_list:
      checkpoint_data = np.append(checkpoint_data, var.eval().flatten())
    np.save(filename, checkpoint_data)

  def load_weights(self, checkpoint_file, weight_initializer=tf.keras.initializers.glorot_normal(),
                   bias_initializer=tf.keras.initializers.Zeros(), net_json_file=None):
    checkpoint_data = np.load(checkpoint_file + '.npy')
    if net_json_file is None:
      net_json_file = checkpoint_file + '.json'
    meta_json = load_json(net_json_file)

    idx = 0
    data_idx = 0

    def initialize_weights_factory(initializer):
      def initialize_weights(name, shape):
        nonlocal idx, data_idx
        if (idx >= len(meta_json)):
          print('load_weights - warning meta_json does not contain data for idx: ' + str(
            idx) + ', using default initializer')
          var = tf.get_variable(name, initializer=initializer(shape))
        else:
          size = 1
          for val in shape:
            size = size * val
          initial_value = np.reshape(checkpoint_data[data_idx:data_idx + size], shape)

          data_idx += size

          var = tf.get_variable(name, initializer=initial_value.astype(np.float32))

          if (var.shape != meta_json[idx]['shape']):
            raise Exception(
              'load_weights - shapes not matching at variable ' + str(var.name) + ': ' + str(var.shape) + ' and ' + str(
                meta_json[idx]['shape']))

          if (var.name != meta_json[idx]['name']):
            print('load_weights - warning: variable names not matching: ' + str(var.name) + ' and ' + str(
              meta_json[idx]['name']))

        idx += 1

        return var

      return initialize_weights

    self.initialize_weights(
      WeightInitializer(initialize_weights_factory(weight_initializer), initialize_weights_factory(bias_initializer)))
