import tensorflow as tf

from .smooth_l1_loss import smooth_l1_loss


class RegressionLossFactory:
  def __init__(self, batch_size=None, scale_by_number_of_positive_anchors=True, scale_by_batch_size=True):
    self.batch_size = batch_size
    self.scale_by_number_of_positive_anchors = scale_by_number_of_positive_anchors
    self.scale_by_batch_size = scale_by_batch_size

  def compile_regression_loss_unscaled(self, y_true, y_pred):
    values = y_true[:, :, :, :2]
    pos_mask = y_true[:, :, :, 2:3]
    return tf.reduce_sum(smooth_l1_loss(values, y_pred) * pos_mask) / (self.batch_size if self.scale_by_batch_size else 1)

  def compile_regression_loss_scaled(self, y_true, y_pred):
    losses = []
    for b in range(0, self.batch_size):
      values = y_true[b:b + 1, :, :, :2]
      pos_mask = y_true[b:b + 1, :, :, 2:3]
      pred_values = y_pred[b:b + 1, :, :, :]

      num_pos = tf.reduce_sum(pos_mask)
      loss = tf.reduce_sum(smooth_l1_loss(values, pred_values) * pos_mask) / tf.maximum(num_pos, 1)
      losses.append(loss)
    return tf.reduce_sum(losses) / (self.batch_size if self.scale_by_batch_size else 1)

  def compile_regression_loss(self, y_true, y_pred):
    if self.scale_by_number_of_positive_anchors:
      return self.compile_regression_loss_scaled(y_true, y_pred)
    return self.compile_regression_loss_unscaled(y_true, y_pred)

  def compile_offsets_loss(self, y_true, pred_offsets):
    return self.compile_regression_loss(y_true, pred_offsets)

  def compile_scales_loss(self, y_true, pred_scales):
    return self.compile_regression_loss(y_true, pred_scales)
