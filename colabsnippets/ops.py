import tensorflow as tf
import numpy as np

def normalize(x, mean_rgb):
  r, g, b = mean_rgb
  shape = np.append(np.array(x.shape[0 : 3]), [1])
  avg_r = tf.fill(shape, r)
  avg_g = tf.fill(shape, g)
  avg_b = tf.fill(shape, b)
  avg_rgb = tf.concat([avg_r, avg_g, avg_b], 3)

  return tf.divide(tf.subtract(x, avg_rgb), 256)

def batch_norm(x, name):
  with tf.variable_scope(name):
    return tf.nn.batch_normalization(x, tf.get_variable('mean'), tf.get_variable('variance'), tf.get_variable('offset'), tf.get_variable('scale'), 1e-3)

def conv2d(x, name, stride, with_batch_norm = False):
  with tf.variable_scope(name):
    out = tf.nn.conv2d(x, tf.get_variable('filter'), stride, 'SAME')
    out = batch_norm(out, 'batch_norm') if with_batch_norm else tf.add(out, tf.get_variable('bias'))
    return out

def depthwise_separable_conv2d(x, name, stride, with_batch_norm = False):
  with tf.variable_scope(name):
    out = tf.nn.separable_conv2d(x, tf.get_variable('depthwise_filter'), tf.get_variable('pointwise_filter'), stride, 'SAME')
    out = batch_norm(out, 'batch_norm') if with_batch_norm else tf.add(out, tf.get_variable('bias'))
    return out

def fully_connected(x, name):
  with tf.variable_scope(name):
    weights = tf.get_variable('weights')
    out = tf.reshape(x, [-1, weights.get_shape().as_list()[0]])
    out = tf.matmul(out, weights)
    out = tf.add(out, tf.get_variable('bias'))
  return out

def dense_block(x, name, is_first_layer = False, is_scale_down = True, use_depthwise_separable_conv2d = True, with_batch_norm = False):
  conv_op = depthwise_separable_conv2d if use_depthwise_separable_conv2d else conv2d
  initial_stride = [1, 2, 2, 1]  if is_scale_down else [1, 1, 1, 1]

  with tf.variable_scope(name):
    if is_first_layer:
      out1 = conv2d(x, 'conv0', initial_stride, with_batch_norm = with_batch_norm)
    else:
      out1 = conv_op(x, 'conv0', initial_stride, with_batch_norm = with_batch_norm)

    in2 = tf.nn.relu(out1)
    out2 = conv_op(in2, 'conv1', [1, 1, 1, 1], with_batch_norm = with_batch_norm)

    in3 = tf.nn.relu(tf.add(out1, out2))
    out3 = conv_op(in3, 'conv2', [1, 1, 1, 1], with_batch_norm = with_batch_norm)

    in4 = tf.nn.relu(tf.add(out1, tf.add(out2, out3)))
    out4 = conv_op(in4, 'conv3', [1, 1, 1, 1], with_batch_norm = with_batch_norm)

    return tf.nn.relu(tf.add(out1, tf.add(out2, tf.add(out3, out4))))

def bottleneck(x, name, stride, is_residual = False):
  with tf.variable_scope(name):
    out = conv2d(x, 'expansion_conv', [1, 1, 1, 1])
    out = depthwise_separable_conv2d(out, 'separable_conv', stride)
    if is_residual:
      out = tf.add(x, out)
    return tf.nn.relu6(out)