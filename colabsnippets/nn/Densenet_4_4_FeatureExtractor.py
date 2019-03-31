import tensorflow as tf

from ..NeuralNetwork import NeuralNetwork
from ..ops import dense_block, normalize

class Densenet_4_4_FeatureExtractor(NeuralNetwork):
  def __init__(self, use_depthwise_separable_conv2d = True, with_batch_norm = False, name = 'densenet_4_4_feature_extractor', channel_multiplier = 1.0, is_scale_down_first_layer = True):
    super().__init__(self.initialize_weights, name = name)
    self.use_depthwise_separable_conv2d = use_depthwise_separable_conv2d
    self.with_batch_norm = with_batch_norm
    self.channel_multiplier = channel_multiplier
    self.is_scale_down_first_layer = is_scale_down_first_layer

  def _create_dense_block_weight_processor(self, weight_processor):
    def _dense_block_weight_processor(cin, cout, name, is_first_layer = False):
      return weight_processor.process_dense_block_weights(
        cin,
        cout,
        name,
        is_first_layer = is_first_layer,
        use_depthwise_separable_conv2d = self.use_depthwise_separable_conv2d,
        with_batch_norm = self.with_batch_norm
      )

    return _dense_block_weight_processor

  def _create_dense_block(self):
    def _dense_block(x, name, is_first_layer = False, is_scale_down = True):
      return dense_block(
        x,
        name,
        is_first_layer = is_first_layer,
        use_depthwise_separable_conv2d = self.use_depthwise_separable_conv2d,
        with_batch_norm = self.with_batch_norm,
        is_scale_down = is_scale_down
      )

    return _dense_block

  def initialize_weights(self, weight_processor, variable_scope = 'feature_extractor'):
    process_dense_block_weights = self._create_dense_block_weight_processor(weight_processor)

    with tf.variable_scope(self.name):
      with tf.variable_scope(variable_scope):
        c0 = int(self.channel_multiplier * 32)
        process_dense_block_weights(3, c0, 'dense0', is_first_layer = True)
        process_dense_block_weights(c0, c0 * 2, 'dense1')
        process_dense_block_weights(c0 * 2, c0 * 4, 'dense2')
        process_dense_block_weights(c0 * 4, c0 * 8, 'dense3')

  def forward(self, batch_tensor, variable_scope = 'feature_extractor'):
    dense_block = self._create_dense_block()

    mean_rgb = [122.782, 117.001, 104.298]
    normalized = normalize(batch_tensor, mean_rgb)

    with tf.variable_scope(self.name, reuse = True):
      with tf.variable_scope(variable_scope):
        out = dense_block(normalized, 'dense0', is_first_layer = True, is_scale_down = self.is_scale_down_first_layer)
        out = dense_block(out, 'dense1')
        out = dense_block(out, 'dense2')
        out = dense_block(out, 'dense3')

      return out