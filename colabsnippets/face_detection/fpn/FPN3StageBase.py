import tensorflow as tf

from .FPNBase import FPNBase
from .generate_anchors import create_anchor
from ...ops import conv2d


class FPN3StageBase(FPNBase):
  def __init__(self, name='fpn3stagebase', anchors=None, stage_filters=None, with_detection_module=True,
               use_minimal_anchors=True):
    self.with_detection_module = with_detection_module
    self.stage_filters = stage_filters
    if use_minimal_anchors:
      stage1_anchors = [create_anchor(i) for i in [16, 32]]
      stage2_anchors = [create_anchor(i) for i in [64, 128]]
      stage3_anchors = [create_anchor(i) for i in [256, 512]]
      anchors = anchors if anchors is not None else [stage1_anchors, stage2_anchors, stage3_anchors]
    else:
      stage1_anchors = [create_anchor(i) for i in [1, 2, 4, 8, 12, 16, 24]]
      stage2_anchors = [create_anchor(i) for i in [8, 16, 24, 32, 64, 96, 128]]
      stage3_anchors = [create_anchor(i) for i in [64, 128, 160, 224, 320, 416, 512]]
      anchors = anchors if anchors is not None else [stage1_anchors, stage2_anchors, stage3_anchors]

    if with_detection_module:
      name += '_ctx'
    if use_minimal_anchors:
      name += '_v2'
    super().__init__(name=name, anchors=anchors, stage_idx_offset=1)

  def init_bottom_up_weights(self, weight_processor):
    raise Exception('FPN3StageBase - init_bottom_up_weights not implemented')

  def bottom_up(self, x):
    raise Exception('FPN3StageBase - bottom_up not implemented')

  def init_context_module_weights(self, weight_processor):
    weight_processor.process_conv_weights(64, 16, 'conv_shrink', filter_size=3)
    weight_processor.process_conv_weights(16, 16, 'conv_1', filter_size=3)
    weight_processor.process_conv_weights(16, 16, 'conv_out_1', filter_size=3)
    weight_processor.process_conv_weights(16, 16, 'conv_out_2', filter_size=3)

  def init_detection_module_weights(self, weight_processor):
    weight_processor.process_conv_weights(64, 32, 'conv_shrink', filter_size=3)
    with tf.variable_scope('ctx'):
      self.init_context_module_weights(weight_processor)

  def initialize_weights(self, weight_processor):
    with tf.variable_scope(self.name):
      with tf.variable_scope('bottom_up'):
        self.init_bottom_up_weights(weight_processor)

      with tf.variable_scope('top_down'):
        weight_processor.process_conv_weights(self.stage_filters[0], 64, 'conv_shrink_1', filter_size=1)
        weight_processor.process_conv_weights(self.stage_filters[1], 64, 'conv_shrink_2', filter_size=1)
        weight_processor.process_conv_weights(self.stage_filters[2], 64, 'conv_shrink_3', filter_size=1)

        weight_processor.process_conv_weights(64, 64, 'conv_anti_aliasing_1', filter_size=3)
        weight_processor.process_conv_weights(64, 64, 'conv_anti_aliasing_2', filter_size=3)

      if self.with_detection_module:
        with tf.variable_scope('det_1'):
          self.init_detection_module_weights(weight_processor)
        with tf.variable_scope('det_2'):
          self.init_detection_module_weights(weight_processor)
        with tf.variable_scope('det_3'):
          self.init_detection_module_weights(weight_processor)

      with tf.variable_scope('classifier'):
        weight_processor.process_conv_weights(64, self.get_num_anchors_per_stage() * 5, 'conv_out_0', filter_size=1)
        weight_processor.process_conv_weights(64, self.get_num_anchors_per_stage() * 5, 'conv_out_1', filter_size=1)
        weight_processor.process_conv_weights(64, self.get_num_anchors_per_stage() * 5, 'conv_out_2', filter_size=1)

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

  def top_down(self, x1, x2, x3, image_size):
    x1 = conv2d(x1, 'conv_shrink_1', [1, 1, 1, 1])
    x2 = conv2d(x2, 'conv_shrink_2', [1, 1, 1, 1])
    x3 = conv2d(x3, 'conv_shrink_3', [1, 1, 1, 1])

    get_stage_size_shape = lambda stage: [self.get_num_cells_for_stage(image_size, stage),
                                          self.get_num_cells_for_stage(image_size, stage)]

    out3 = out = x3
    out2 = out = tf.add(tf.image.resize_images(out, get_stage_size_shape(1)), x2)
    out1 = out = tf.add(tf.image.resize_images(out, get_stage_size_shape(0)), x1)

    out2 = tf.nn.relu(conv2d(out2, 'conv_anti_aliasing_2', [1, 1, 1, 1]))
    out1 = tf.nn.relu(conv2d(out1, 'conv_anti_aliasing_1', [1, 1, 1, 1]))

    return out1, out2, out3

  def forward(self, batch_tensor, batch_size, image_size, out_num_cells=None):
    def normalize(x):
      return tf.divide(tf.subtract(x, 123), 256)

    out = normalize(batch_tensor)

    with tf.variable_scope(self.name, reuse=True):
      with tf.variable_scope('bottom_up', reuse=True):
        out1, out2, out3 = self.bottom_up(out)

      with tf.variable_scope('top_down', reuse=True):
        out1, out2, out3 = self.top_down(out1, out2, out3, image_size)

      if self.with_detection_module:
        with tf.variable_scope('det_1', reuse=True):
          out1 = self.detection_module(out1)
        with tf.variable_scope('det_2', reuse=True):
          out2 = self.detection_module(out2)
        with tf.variable_scope('det_3', reuse=True):
          out3 = self.detection_module(out3)

      with tf.variable_scope('classifier', reuse=True):
        out1 = conv2d(out1, 'conv_out_0', [1, 1, 1, 1])
        out2 = conv2d(out2, 'conv_out_1', [1, 1, 1, 1])
        out3 = conv2d(out3, 'conv_out_2', [1, 1, 1, 1])

        out1 = self.coords_and_scores(out1, self.get_num_cells_for_stage(image_size, 0), batch_size)
        out2 = self.coords_and_scores(out2, self.get_num_cells_for_stage(image_size, 1), batch_size)
        out3 = self.coords_and_scores(out3, self.get_num_cells_for_stage(image_size, 2), batch_size)

      return out1, out2, out3
