import tensorflow as tf

from .FPN3StageBase import FPN3StageBase

from ...ops import conv2d, reduction_block, main_block

class FPN3StageMD(FPN3StageBase):
  def __init__(self, name = 'fpn3stagemd', with_detection_module = True):
    super().__init__(name = name, stage_filters = [128, 256, 512], with_detection_module = with_detection_module)

  def init_bottom_up_weights(self, weight_processor):
    weight_processor.process_conv_weights(3, 32, 'conv_in', filter_size = 3)

    weight_processor.process_reduction_block_weights(32, 64, 'reduction_block_0')
    weight_processor.process_main_block_weights(64, 'main_block_0_0')

    weight_processor.process_reduction_block_weights(64, 128, 'reduction_block_1')
    weight_processor.process_main_block_weights(128, 'main_block_1_0')
    weight_processor.process_main_block_weights(128, 'main_block_1_1')

    weight_processor.process_reduction_block_weights(128, 256, 'reduction_block_2')
    weight_processor.process_main_block_weights(256, 'main_block_2_0')
    weight_processor.process_main_block_weights(256, 'main_block_2_1')

    weight_processor.process_reduction_block_weights(256, 512, 'reduction_block_3')
    weight_processor.process_main_block_weights(512, 'main_block_3_0')
    weight_processor.process_main_block_weights(512, 'main_block_3_1')
    weight_processor.process_main_block_weights(512, 'main_block_3_2')

  def bottom_up(self, x):
    out = tf.nn.relu(conv2d(x, 'conv_in', [1, 2, 2, 1]))

    out = reduction_block(out, 'reduction_block_0', is_activate_input = False)
    out = main_block(out, 'main_block_0_0')

    out = reduction_block(out, 'reduction_block_1')
    out = main_block(out, 'main_block_1_0')
    out1 = out = main_block(out, 'main_block_1_1')

    out = reduction_block(out, 'reduction_block_2')
    out = main_block(out, 'main_block_2_0')
    out2 = out = main_block(out, 'main_block_2_1')

    out = reduction_block(out, 'reduction_block_3')
    out = main_block(out, 'main_block_3_0')
    out = main_block(out, 'main_block_3_1')
    out3 = out = main_block(out, 'main_block_3_2')

    return out1, out2, out3