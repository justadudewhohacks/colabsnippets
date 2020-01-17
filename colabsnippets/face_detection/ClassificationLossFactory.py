import tensorflow as tf

from colabsnippets.face_detection import focal_loss


class ClassificationLossFactory:
  def __init__(self, batch_size=None, scale_by_number_of_positive_anchors=True, scale_by_batch_size=True,
               use_soft_negative_mask=True, hnm_neg_pos_ratio=3.0, loss_type='focal', hnm_num_cells=None):
    if loss_type != 'focal' and loss_type != 'hnm':
      raise Exception('ClassificationLossFactory - invalid loss_type: ' + str(loss_type)
                      + ', valid types are focal | hnm')
    if loss_type == 'hnm' and hnm_num_cells is None:
      raise Exception('ClassificationLossFactory - for hnm parameter hnm_num_cells is required')
    self.batch_size = batch_size
    self.scale_by_number_of_positive_anchors = scale_by_number_of_positive_anchors
    self.scale_by_batch_size = scale_by_batch_size
    self.use_soft_negative_mask = use_soft_negative_mask
    self.hnm_neg_pos_ratio = hnm_neg_pos_ratio
    self.loss_type = loss_type
    self.hnm_num_cells = hnm_num_cells

  def compile_classification_loss_scaled(self, y_true, y_pred, compute_loss):
    losses = []
    for b in range(0, self.batch_size):
      pos_mask = y_true[b:b + 1, :, :, 0:1]
      neg_mask = y_true[b:b + 1, :, :, 1:2]
      neg_mask = neg_mask if self.use_soft_negative_mask else 1 - pos_mask
      pred_scores = y_pred[b:b + 1, :, :, :]
      num_pos = tf.reduce_sum(pos_mask)

      loss = compute_loss(pos_mask, neg_mask, pred_scores, num_pos)
      loss = tf.reshape(loss, [-1])
      loss = tf.reduce_sum(loss) / tf.maximum(num_pos, 1)
      losses.append(loss)
    return tf.reduce_sum(losses) / (self.batch_size if self.scale_by_batch_size else 1)

  def compile_object_loss_scaled(self, y_true, y_pred):
    def compute_loss(pos_mask, _, pred_scores, __):
      if self.loss_type == 'focal':
        return pos_mask * focal_loss(pred_scores)
      elif self.loss_type == 'hnm':
        return pos_mask * (1 - pred_scores)
      raise Exception('unknown loss_type: ' + str(self.loss_type))

    return self.compile_classification_loss_scaled(y_true, y_pred, compute_loss)

  def compile_no_object_loss_scaled(self, y_true, y_pred):
    def compute_loss(_, neg_mask, pred_scores, num_pos):
      if self.loss_type == 'focal':
        return neg_mask * focal_loss(1 - pred_scores)
      elif self.loss_type == 'hnm':
        num_neg = tf.minimum(num_pos * self.hnm_neg_pos_ratio, self.hnm_num_cells * self.hnm_num_cells)
        loss = (neg_mask * pred_scores)
        loss = tf.reshape(loss, [-1])
        loss, _ = tf.nn.top_k(loss, k=tf.cast(num_neg, tf.int32))
        return loss

    return self.compile_classification_loss_scaled(y_true, y_pred, compute_loss)
