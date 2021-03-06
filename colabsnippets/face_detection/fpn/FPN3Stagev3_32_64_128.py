import tensorflow as tf

from colabsnippets.face_detection.fpn import FPN3StageBase
from colabsnippets.ops import conv2d, reduction_block, main_block


class FPN3Stagev3_32_64_128(FPN3StageBase):
  def __init__(self, name='fpn3stagev3_32_64_128'):
    super().__init__(name=name, stage_filters=[48, 64, 128], out_channels=64,
                     with_detection_module=True, use_minimal_anchors=True, net_suffix="",
                     with_batch_norm=False, stage_idx_offset=2)

  def init_bottom_up_weights(self, weight_processor):
    weight_processor.process_conv_weights(3, 8, 'conv_in', filter_size=3)
    weight_processor.process_reduction_block_weights(8, 16, 'reduction_block_0')
    weight_processor.process_reduction_block_weights(16, 24, 'reduction_block_1')
    weight_processor.process_reduction_block_weights(24, 48, 'reduction_block_2')
    weight_processor.process_reduction_block_weights(48, 64, 'reduction_block_3')
    weight_processor.process_main_block_weights(64, 'main_block_3_0')
    weight_processor.process_reduction_block_weights(64, 128, 'reduction_block_4')
    weight_processor.process_main_block_weights(128, 'main_block_4_0')

  def bottom_up(self, x):
    out = tf.nn.relu(conv2d(x, 'conv_in', [1, 2, 2, 1]))
    out = reduction_block(out, 'reduction_block_0', is_activate_input=False)
    out = reduction_block(out, 'reduction_block_1')
    out1 = out = reduction_block(out, 'reduction_block_2')
    out = reduction_block(out, 'reduction_block_3')
    out2 = out = main_block(out, 'main_block_3_0')
    out = reduction_block(out, 'reduction_block_4')
    out3 = main_block(out, 'main_block_4_0')

    return out1, out2, out3
