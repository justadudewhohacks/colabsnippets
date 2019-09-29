import tensorflow as tf


class WeightInitializer:
  def __init__(self, weight_initializer, bias_initializer):
    self.weight_initializer = weight_initializer
    self.bias_initializer = bias_initializer

  def process_batch_norm_weights(self, channels, name):
    with tf.variable_scope(name):
      tf.get_variable('mean', shape=[channels], initializer=tf.keras.initializers.Zeros)
      tf.get_variable('variance', shape=[channels], initializer=tf.keras.initializers.Ones)
      tf.get_variable('offset', shape=[channels], initializer=tf.keras.initializers.Zeros)
      tf.get_variable('scale', shape=[channels], initializer=tf.keras.initializers.Ones)

  def process_conv_weights(self, channels_in, channels_out, name, filter_size=3, with_batch_norm=False):
    with tf.variable_scope(name):
      self.weight_initializer('filter', [filter_size, filter_size, channels_in, channels_out])
      if with_batch_norm:
        self.process_batch_norm_weights(channels_out, 'batch_norm')
      else:
        self.bias_initializer('bias', [channels_out])

  def process_fc_weights(self, channels_in, channels_out, name):
    with tf.variable_scope(name):
      self.weight_initializer('weights', [channels_in, channels_out])
      self.bias_initializer('bias', [channels_out])

  def process_depthwise_separable_conv2d_weights(self, channels_in, channels_out, name, with_batch_norm=False):
    with tf.variable_scope(name):
      self.weight_initializer('depthwise_filter', [3, 3, channels_in, 1])
      self.weight_initializer('pointwise_filter', [1, 1, channels_in, channels_out])
      if with_batch_norm:
        self.process_batch_norm_weights(channels_out, 'batch_norm')
      else:
        self.bias_initializer('bias', [channels_out])

  def process_dense_block_weights(self, channels_in, channels_out, name, is_first_layer=False,
                                  use_depthwise_separable_conv2d=True, with_batch_norm=False):
    conv_weight_processor = self.process_depthwise_separable_conv2d_weights if use_depthwise_separable_conv2d else self.process_conv_weights
    conv0_weight_processor = self.process_conv_weights if is_first_layer else conv_weight_processor

    with tf.variable_scope(name):
      conv0_weight_processor(channels_in, channels_out, 'conv0', with_batch_norm=with_batch_norm)
      conv_weight_processor(channels_out, channels_out, 'conv1', with_batch_norm=with_batch_norm)
      conv_weight_processor(channels_out, channels_out, 'conv2', with_batch_norm=with_batch_norm)
      conv_weight_processor(channels_out, channels_out, 'conv3', with_batch_norm=with_batch_norm)

  def process_reduction_block_weights(self, channels_in, channels_out, name, with_batch_norm=False):
    with tf.variable_scope(name):
      self.process_depthwise_separable_conv2d_weights(channels_in, channels_out, 'separable_conv0',
                                                      with_batch_norm=with_batch_norm)
      self.process_depthwise_separable_conv2d_weights(channels_out, channels_out, 'separable_conv1',
                                                      with_batch_norm=with_batch_norm)
      self.process_conv_weights(channels_in, channels_out, 'expansion_conv', filter_size=1,
                                with_batch_norm=with_batch_norm)

  def process_main_block_weights(self, channels, name, with_batch_norm=False):
    with tf.variable_scope(name):
      self.process_depthwise_separable_conv2d_weights(channels, channels, 'separable_conv0',
                                                      with_batch_norm=with_batch_norm)
      self.process_depthwise_separable_conv2d_weights(channels, channels, 'separable_conv1',
                                                      with_batch_norm=with_batch_norm)
      self.process_depthwise_separable_conv2d_weights(channels, channels, 'separable_conv2',
                                                      with_batch_norm=with_batch_norm)
