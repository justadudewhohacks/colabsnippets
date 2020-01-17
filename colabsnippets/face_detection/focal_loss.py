import tensorflow as tf


def focal_loss(p, gamma=2.0):
  return -(1 - p) ** gamma * tf.log(tf.maximum(0.00001, p))
