import tensorflow as tf

from .FPN3StageBase import FPN3StageBase
from ...ops import conv2d, depthwise_separable_conv2d, reduction_block, main_block


class FPN3Stagev2Base(FPN3StageBase):
  def __init__(self, name='fpn3stagev2base', with_batch_norm=True, stage_filters=None, channel_multiplier=1,
               is_extended_first_layer=False):
    super().__init__(name=name, stage_filters=stage_filters,
                     with_detection_module=True, use_minimal_anchors=True, net_suffix="",
                     with_batch_norm=with_batch_norm)
    self.with_batch_norm = with_batch_norm
    self.channel_multiplier = channel_multiplier
    self.is_extended_first_layer = is_extended_first_layer

  def init_bottom_up_weights(self, weight_processor):
    c = self.channel_multiplier * 8
    c0 = c * (2 if self.is_extended_first_layer else 1)
    weight_processor.process_conv_weights(3, c0, 'conv_in', filter_size=3, with_batch_norm=self.with_batch_norm)
    weight_processor.process_reduction_block_weights(c0, c0*2, 'reduction_block_0', with_batch_norm=self.with_batch_norm)
    weight_processor.process_reduction_block_weights(c0*2, c0*4, 'reduction_block_1', with_batch_norm=self.with_batch_norm)
    if not self.is_extended_first_layer:
      weight_processor.process_depthwise_separable_conv2d_weights(c*4, c*8, 'separable_conv1',
                                                                  with_batch_norm=self.with_batch_norm)
    weight_processor.process_main_block_weights(c*8, 'main_block_1_0', with_batch_norm=self.with_batch_norm)
    weight_processor.process_reduction_block_weights(c*8, c*16, 'reduction_block_2', with_batch_norm=self.with_batch_norm)
    weight_processor.process_main_block_weights(c*16, 'main_block_2_0', with_batch_norm=self.with_batch_norm)
    weight_processor.process_reduction_block_weights(c*16, c*32, 'reduction_block_3',
                                                     with_batch_norm=self.with_batch_norm)
    weight_processor.process_main_block_weights(c*32, 'main_block_3_0', with_batch_norm=self.with_batch_norm)

  def bottom_up(self, x):
    out = tf.nn.relu(conv2d(x, 'conv_in', [1, 2, 2, 1], with_batch_norm=self.with_batch_norm))
    out = reduction_block(out, 'reduction_block_0', is_activate_input=False, with_batch_norm=self.with_batch_norm)
    out = reduction_block(out, 'reduction_block_1', with_batch_norm=self.with_batch_norm)
    if not self.is_extended_first_layer:
      out = depthwise_separable_conv2d(tf.nn.relu(out), 'separable_conv1', [1, 1, 1, 1],
                                       with_batch_norm=self.with_batch_norm)
    out1 = out = main_block(out, 'main_block_1_0', with_batch_norm=self.with_batch_norm)
    out = reduction_block(out, 'reduction_block_2', with_batch_norm=self.with_batch_norm)
    out2 = out = main_block(out, 'main_block_2_0', with_batch_norm=self.with_batch_norm)
    out = reduction_block(out, 'reduction_block_3', with_batch_norm=self.with_batch_norm)
    out3 = main_block(out, 'main_block_3_0', with_batch_norm=self.with_batch_norm)

    return out1, out2, out3
