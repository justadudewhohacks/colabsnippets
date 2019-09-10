import tensorflow as tf

def focal_loss(p, is_gt, gamma = 2):
  pt = p if is_gt else (1 - p)
  return -(1 - pt)**gamma * tf.log(tf.maximum(0.00001, pt))