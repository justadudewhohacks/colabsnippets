import tensorflow as tf

from .FPNBase import FPNBase
from .generate_anchors import create_anchor
from ...ops import conv2d


class FPN2StageBase(FPNBase):
  def __init__(self, name='fpn2stagebase', stage_filters=None, out_channels=64):
    self.stage_filters = stage_filters
    self.out_channels = out_channels
    stage1_anchors = [create_anchor(i) for i in [32, 64, 128]]
    stage2_anchors = [create_anchor(i) for i in [256, 512]]
    super().__init__(name=name, anchors=[stage1_anchors, stage2_anchors], stage_idx_offset=3)

  def init_bottom_up_weights(self, weight_processor):
    raise Exception('FPN2StageBase - init_bottom_up_weights not implemented')

  def bottom_up(self, x):
    raise Exception('FPN2StageBase - bottom_up not implemented')

  def init_context_module_weights(self, weight_processor):
    c = int(self.out_channels / 4)
    weight_processor.process_conv_weights(self.out_channels, c, 'conv_shrink', filter_size=3)
    weight_processor.process_conv_weights(c, c, 'conv_1', filter_size=3)
    weight_processor.process_conv_weights(c, c, 'conv_out_1', filter_size=3)
    weight_processor.process_conv_weights(c, c, 'conv_out_2', filter_size=3)

  def init_detection_module_weights(self, weight_processor):
    weight_processor.process_conv_weights(self.out_channels, int(self.out_channels / 2), 'conv_shrink', filter_size=3)
    with tf.variable_scope('ctx'):
      self.init_context_module_weights(weight_processor)

  def initialize_weights(self, weight_processor, reuse=False):
    with tf.variable_scope(self.name, reuse=reuse):
      with tf.variable_scope('bottom_up'):
        self.init_bottom_up_weights(weight_processor)

      with tf.variable_scope('top_down'):
        weight_processor.process_conv_weights(self.stage_filters[0], self.out_channels, 'conv_shrink_1', filter_size=1)
        weight_processor.process_conv_weights(self.stage_filters[1], self.out_channels, 'conv_shrink_2', filter_size=1)

        weight_processor.process_conv_weights(self.out_channels, self.out_channels, 'conv_anti_aliasing_1',
                                              filter_size=3)

      with tf.variable_scope('det_1'):
        self.init_detection_module_weights(weight_processor)
      with tf.variable_scope('det_2'):
        self.init_detection_module_weights(weight_processor)

      with tf.variable_scope('classifier'):
        weight_processor.process_conv_weights(self.out_channels, self.get_num_anchors_for_stage(0) * 5, 'conv_out_0',
                                              filter_size=1)
        weight_processor.process_conv_weights(self.out_channels, self.get_num_anchors_for_stage(1) * 5, 'conv_out_1',
                                              filter_size=1)

  def context_module(self, x):
    shrink = out = tf.nn.relu(conv2d(x, 'conv_shrink', [1, 1, 1, 1]))
    out = tf.nn.relu(conv2d(out, 'conv_1', [1, 1, 1, 1]))
    out1 = conv2d(out, 'conv_out_1', [1, 1, 1, 1])
    out2 = conv2d(shrink, 'conv_out_2', [1, 1, 1, 1])
    return out1, out2

  def detection_module(self, x):
    with tf.variable_scope('ctx', reuse=True):
      ctx_out_1, ctx_out_2 = self.context_module(x)

    shrink = conv2d(x, 'conv_shrink', [1, 1, 1, 1])
    out = tf.nn.relu(tf.concat([shrink, ctx_out_1, ctx_out_2], axis=3))
    return out

  def top_down(self, x1, x2, image_size):
    x1 = conv2d(x1, 'conv_shrink_1', [1, 1, 1, 1])
    x2 = conv2d(x2, 'conv_shrink_2', [1, 1, 1, 1])

    get_stage_size_shape = lambda stage: [self.get_num_cells_for_stage(image_size, stage),
                                          self.get_num_cells_for_stage(image_size, stage)]

    out2 = out = x2
    out1 = out = tf.add(tf.image.resize_images(out, get_stage_size_shape(0)), x1)

    out1 = tf.nn.relu(conv2d(out1, 'conv_anti_aliasing_1', [1, 1, 1, 1]))

    return out1, out2

  def forward(self, batch_tensor, batch_size, image_size, out_num_cells=None):
    def normalize(x):
      return tf.divide(tf.subtract(x, 123), 256)

    out = normalize(batch_tensor)

    with tf.variable_scope(self.name, reuse=True):
      with tf.variable_scope('bottom_up', reuse=True):
        out1, out2 = self.bottom_up(out)

      with tf.variable_scope('top_down', reuse=True):
        out1, out2 = self.top_down(out1, out2, image_size)

      with tf.variable_scope('det_1', reuse=True):
        out1 = self.detection_module(out1)
      with tf.variable_scope('det_2', reuse=True):
        out2 = self.detection_module(out2)

      with tf.variable_scope('classifier', reuse=True):
        out1 = conv2d(out1, 'conv_out_0', [1, 1, 1, 1])
        out2 = conv2d(out2, 'conv_out_1', [1, 1, 1, 1])

        out1 = self.coords_and_scores(out1, image_size, batch_size, 0)
        out2 = self.coords_and_scores(out2, image_size, batch_size, 1)

      return out1, out2
