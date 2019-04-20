import tensorflow as tf
import numpy as np

def create_fake_input_tensor(image_size):
  return np.zeros([1, image_size, image_size, 3], np.float32)

def test_net(net, x, expected_output_shapes):
  with tf.Session() as sess:
    net.init_trainable_weights()
    sess.run(tf.global_variables_initializer())

    res = net.forward(x)

    if isinstance(expected_output_shapes, tuple):
      for idx, r in enumerate(res):
        np.testing.assert_array_equal(expected_output_shapes[idx], r.shape)
    else:
      np.testing.assert_array_equal(expected_output_shapes, res.shape)

  tf.reset_default_graph()