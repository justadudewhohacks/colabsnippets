import tensorflow as tf

from ..nn import XceptionTiny
from ..ops import normalize, conv2d, depthwise_separable_conv2d, fully_connected

class AgeGenderXceptionTiny(XceptionTiny):
  def __init__(self, name = 'age_gender_xception_tiny', num_blocks = 8):
    super().__init__(name = name)
    self.num_blocks = num_blocks

  def initialize_weights(self, weight_processor):
    super().initialize_weights(weight_processor)

    with tf.variable_scope(self.name):
      with tf.variable_scope('classifier'):
        weight_processor.process_fc_weights(512, 1, 'fc_age')
        weight_processor.process_fc_weights(512, 2, 'fc_gender')

  def forward(self, batch_tensor):
    out = super().forward(batch_tensor)

    with tf.variable_scope(self.name, reuse = True):
      with tf.variable_scope('classifier'):
        out = tf.nn.avg_pool(out, [1, 7, 7, 1], [1, 2, 2, 1], 'VALID')
        out_age = fully_connected(out, 'fc_age')
        out_age = tf.reshape(out_age, [out_age.shape[0]])
        out_gender = fully_connected(out, 'fc_gender')

    return out_age, out_gender