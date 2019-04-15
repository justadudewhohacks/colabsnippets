import tensorflow as tf

from ..NeuralNetwork import NeuralNetwork
from ..ops import bottleneck, normalized

class MobilenetV2(NeuralNetwork):
  def __init__(self, name = 'mobilenetv2'):
    super().__init__(self.initialize_weights, name = name)

  def initialize_weights(self, weight_processor):
    def process_bottleneck_weights(channels_in, channels_out, expansion_factor, name):
      with tf.variable_scope(name):
        channels_expand = channels_in * expansion_factor
        weight_processor.process_conv_weights(channels_in, channels_expand, 'expansion_conv', filter_size = 1)
        weight_processor.process_depthwise_separable_conv2d_weights(channels_expand, channels_out, 'separable_conv')

    with tf.variable_scope(self.name):
      weight_processor.process_conv_weights(3, 32, 'conv_in', filter_size = 3)
      process_bottleneck_weights(32, 16, 1, 'bottleneck0/n0')
      process_bottleneck_weights(16, 24, 6, 'bottleneck1/n0')
      process_bottleneck_weights(24, 24, 6, 'bottleneck1/n1')
      process_bottleneck_weights(24, 24, 6, 'bottleneck1/n2')
      process_bottleneck_weights(24, 32, 6, 'bottleneck2/n0')
      process_bottleneck_weights(32, 32, 6, 'bottleneck2/n1')
      process_bottleneck_weights(32, 32, 6, 'bottleneck2/n2')
      process_bottleneck_weights(32, 64, 6, 'bottleneck3/n0')
      process_bottleneck_weights(64, 64, 6, 'bottleneck3/n1')
      process_bottleneck_weights(64, 64, 6, 'bottleneck3/n2')
      process_bottleneck_weights(64, 128, 6, 'bottleneck4/n0')
      process_bottleneck_weights(128, 128, 6, 'bottleneck4/n1')
      process_bottleneck_weights(128, 128, 6, 'bottleneck4/n2')
      weight_processor.process_conv_weights(128, 512, 'conv_expand', filter_size = 1)
      weight_processor.process_conv_weights(512, 1, 'conv_age_out', filter_size = 1)

  def forward(self, batch_tensor):
    mean_rgb = [122.782, 117.001, 104.298]
    normalized = normalize(batch_tensor, mean_rgb)

    with tf.variable_scope(self.name, reuse = True):
      # initial stride of 1 (112x112 input) instead of 2 (224x224 input)
      out = tf.nn.relu6(conv2d(normalized, 'conv_in', [1, 1, 1, 1]))
      out = bottleneck(out, 'bottleneck0/n0', [1, 1, 1, 1])
      out = bottleneck(out, 'bottleneck1/n0', [1, 2, 2, 1])
      out = bottleneck(out, 'bottleneck1/n1', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck1/n2', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck2/n0', [1, 2, 2, 1])
      out = bottleneck(out, 'bottleneck2/n1', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck2/n2', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck3/n0', [1, 2, 2, 1])
      out = bottleneck(out, 'bottleneck3/n1', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck3/n2', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck4/n0', [1, 2, 2, 1])
      out = bottleneck(out, 'bottleneck4/n1', [1, 1, 1, 1], True)
      out = bottleneck(out, 'bottleneck4/n2', [1, 1, 1, 1], True)
      out = tf.nn.relu6(conv2d(out, 'conv_expand', [1, 1, 1, 1]))
      out = tf.nn.avg_pool(out, [1, 7, 7, 1], [1, 2, 2, 1], 'VALID')
      out = conv2d(out, 'conv_age_out', [1, 1, 1, 1])

    out = tf.reshape(out, [out.shape[0]])
    return out