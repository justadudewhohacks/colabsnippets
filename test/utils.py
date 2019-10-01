import os

import tensorflow as tf
import numpy as np

def ensure_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

def remove_if_exists(file):
  if os.path.exists(file):
    os.remove(file)


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
  tf.reset_default_graph()
  ensure_dir('./tmp')
  net_weights_file = './tmp/' + net.name + '.npy'
  meta_json_file = './tmp/' + net.name + '.json'

  try:
    var_names_init = []
    vars_init = []
    var_names_loaded = []
    vars_loaded = []
    all_vars = []

    with tf.Session() as sess:
      net.init_trainable_weights()
      sess.run(tf.global_variables_initializer())
      for var in tf.global_variables():
        all_vars.append(var.name)
      for var in net.get_net_vars_in_initialization_order():
        var_names_init.append(var.name)
        vars_init.append(var.eval())

      net.save_meta_json('./tmp/' + net.name)
      net.save_weights('./tmp/' + net.name)

    tf.reset_default_graph()
    with tf.Session() as sess:
      net.load_weights('./tmp/' + net.name, net_json_file=meta_json_file)
      sess.run(tf.global_variables_initializer())
      for var in net.get_net_vars_in_initialization_order():
        var_names_loaded.append(var.name)
        vars_loaded.append(var.eval())

      np.testing.assert_equal(len(vars_init), len(all_vars))
      np.testing.assert_equal(len(vars_loaded), len(all_vars))
      for idx in range(0, len(vars_loaded)):
        np.testing.assert_array_equal(vars_init[idx], vars_loaded[idx])

      # TODO check output shape
        forward()
    tf.reset_default_graph()

  finally:
    tf.reset_default_graph()
    remove_if_exists(net_weights_file)
    remove_if_exists(meta_json_file)