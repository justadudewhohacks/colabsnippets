import tensorflow as tf
import numpy as np

from ..nn import Densenet_4_4_FeatureExtractor
from ..ops import fully_connected

class Densenet_4_4_DEX(Densenet_4_4_FeatureExtractor):
  def __init__(self, use_depthwise_separable_conv2d = True, with_batch_norm = False, name = 'densenet_4_4_dex'):
    super().__init__(
      use_depthwise_separable_conv2d = use_depthwise_separable_conv2d,
      with_batch_norm = with_batch_norm,
      name = name
    )

  def initialize_weights(self, weight_processor):
    super().initialize_weights(weight_processor)
    with tf.variable_scope(self.name):
      with tf.variable_scope('classifier'):
        weight_processor.process_fc_weights(256, 101, 'fc_age')

  def forward(self, batch_tensor):
    out = super().forward(batch_tensor)
    with tf.variable_scope(self.name, reuse = True):
      with tf.variable_scope('classifier'):
        out = tf.nn.avg_pool(out, [1, 7, 7, 1], [1, 2, 2, 1], 'VALID')
        out = fully_connected(out, 'fc_age')
        out = tf.reduce_sum(tf.multiply(tf.nn.softmax(out), np.arange(101)), axis = 1)

    return out
