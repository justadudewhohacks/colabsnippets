import tensorflow as tf

from .Densenet_4_4_FeatureExtractor import Densenet_4_4_FeatureExtractor
from ..ops import dense_block

class Densenet_4_5_FeatureExtractor(Densenet_4_4_FeatureExtractor):
  def __init__(self, use_depthwise_separable_conv2d = True, with_batch_norm = False, name = 'densenet_4_5_feature_extractor', channel_multiplier = 1.0):
    super().__init__(
      use_depthwise_separable_conv2d = use_depthwise_separable_conv2d,
      with_batch_norm = with_batch_norm,
      name = name,
      channel_multiplier = channel_multiplier,
      is_scale_down_first_layer = False
    )

  def initialize_weights(self, weight_processor, variable_scope = 'feature_extractor'):
    process_dense_block_weights = super()._create_dense_block_weight_processor(weight_processor)

    super().initialize_weights(weight_processor, variable_scope = variable_scope)
    with tf.variable_scope(self.name):
      with tf.variable_scope(variable_scope):
        c0 = int(self.channel_multiplier * 32)
        process_dense_block_weights(c0 * 8, c0 * 16, 'dense4')

  def forward(self, batch_tensor, variable_scope = 'feature_extractor'):
    dense_block = super()._create_dense_block()

    out = super().forward(batch_tensor, variable_scope = variable_scope)
    with tf.variable_scope(self.name, reuse = True):
      with tf.variable_scope(variable_scope):
        out = dense_block(out, 'dense4')

      return out
