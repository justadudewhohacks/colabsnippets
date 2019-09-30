import os

import tensorflow as tf
import numpy as np

def create_fake_input_tensor(image_size):
  return np.zeros([1, image_size, image_size, 3], np.float32)

def assert_output_shape_matches(res, expected_output_shapes):
  if isinstance(expected_output_shapes, tuple):
    for idx, r in enumerate(res):
      np.testing.assert_array_equal(expected_output_shapes[idx], r.shape)
  else:
    np.testing.assert_array_equal(expected_output_shapes, res.shape)

def test_net(net, x, expected_output_shapes):
  with tf.Session() as sess:
    net.init_trainable_weights()
    sess.run(tf.global_variables_initializer())
    assert_output_shape_matches(net.forward(x), expected_output_shapes)

  tf.reset_default_graph()

def test_net_save_load_forward(net, forward):
  if not os.path.exists('./tmp'):
    os.mkdir('./tmp')

  net_weights_file = './tmp/' + net.name + '.npy'
  meta_json_file = './tmp/' + net.name + '.json'

  with tf.Session() as sess:
    net.init_trainable_weights()
    sess.run(tf.global_variables_initializer())
    net.save_meta_json('./tmp/' + net.name)
    net.save_weights('./tmp/' + net.name)
  tf.reset_default_graph()

  with tf.Session():
    net.load_weights('./tmp/' + net.name, net_json_file=meta_json_file)
    # TODO check output shape
    forward()
  tf.reset_default_graph()

  os.remove(net_weights_file)
  os.remove(meta_json_file)