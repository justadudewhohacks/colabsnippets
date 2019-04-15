import tensorflow as tf

from ..NeuralNetwork import NeuralNetwork
from ..ops import normalize, conv2d, depthwise_separable_conv2d, fully_connected

class XceptionTiny(NeuralNetwork):
  def __init__(self, name = 'xception_tiny', num_blocks = 8):
    super().__init__(self.initialize_weights, name = name)
    self.num_blocks = num_blocks

  def initialize_weights(self, weight_processor):
    def process_reduction_block_weights(channels_in, channels_out, name):
      with tf.variable_scope(name):
        weight_processor.process_depthwise_separable_conv2d_weights(channels_in, channels_out, 'separable_conv0')
        weight_processor.process_depthwise_separable_conv2d_weights(channels_out, channels_out, 'separable_conv1')
        weight_processor.process_conv_weights(channels_in, channels_out, 'expansion_conv', filter_size = 1)

    def process_main_block_weights(channels, name):
      with tf.variable_scope(name):
        weight_processor.process_depthwise_separable_conv2d_weights(channels, channels, 'separable_conv0')
        weight_processor.process_depthwise_separable_conv2d_weights(channels, channels, 'separable_conv1')
        weight_processor.process_depthwise_separable_conv2d_weights(channels, channels, 'separable_conv2')

    with tf.variable_scope(self.name):
      with tf.variable_scope('entry_flow'):
        weight_processor.process_conv_weights(3, 32, 'conv_in', filter_size = 3)
        process_reduction_block_weights(32, 64, 'reduction_block_0')
        process_reduction_block_weights(64, 128, 'reduction_block_1')

      with tf.variable_scope('middle_flow'):
        for block_num in range(0, self.num_blocks):
          process_main_block_weights(128, 'main_block_' + str(block_num))

      with tf.variable_scope('exit_flow'):
        process_reduction_block_weights(128, 256, 'reduction_block')
        weight_processor.process_depthwise_separable_conv2d_weights(256, 512, 'separable_conv')

      with tf.variable_scope('classifier'):
        weight_processor.process_fc_weights(512, 1, 'fc_age')

  def reduction_block(self, x, name, is_activate_input = True):
    out = x
    with tf.variable_scope(name):
      out = tf.nn.relu(out) if is_activate_input else out
      out = depthwise_separable_conv2d(out, 'separable_conv0', [1, 1, 1, 1])
      out = depthwise_separable_conv2d(tf.nn.relu(out), 'separable_conv1', [1, 1, 1, 1])
      out = tf.nn.max_pool(out, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
      out = tf.add(out, conv2d(x, 'expansion_conv', [1, 2, 2, 1]))
      return out

  def main_block(self, x, name):
    out = x
    with tf.variable_scope(name):
      out = depthwise_separable_conv2d(tf.nn.relu(out), 'separable_conv0', [1, 1, 1, 1])
      out = depthwise_separable_conv2d(tf.nn.relu(out), 'separable_conv1', [1, 1, 1, 1])
      out = depthwise_separable_conv2d(tf.nn.relu(out), 'separable_conv2', [1, 1, 1, 1])
      out = tf.add(out, x)
      return out

  def forward(self, batch_tensor):
    mean_rgb = [122.782, 117.001, 104.298]
    out = normalize(batch_tensor, mean_rgb)

    with tf.variable_scope(self.name, reuse = True):
      with tf.variable_scope('entry_flow'):
        out = tf.nn.relu(conv2d(out, 'conv_in', [1, 2, 2, 1]))
        out = self.reduction_block(out, 'reduction_block_0', is_activate_input = False)
        out = self.reduction_block(out, 'reduction_block_1')

      with tf.variable_scope('middle_flow'):
        for block_num in range(0, self.num_blocks):
          out = self.main_block(out, 'main_block_' + str(block_num))

      with tf.variable_scope('exit_flow'):
        out = self.reduction_block(out, 'reduction_block')
        out = tf.nn.relu(depthwise_separable_conv2d(out, 'separable_conv', [1, 1, 1, 1]))

      with tf.variable_scope('classifier'):
        out = tf.nn.avg_pool(out, [1, 7, 7, 1], [1, 2, 2, 1], 'VALID')
        out = fully_connected(out, 'fc_age')
        out = tf.reshape(out, [out.shape[0]])

    return out