import unittest

import tensorflow as tf
import numpy as np

from colabsnippets.nn import (
  Densenet_4_4_FeatureExtractor,
  Densenet_4_5_FeatureExtractor,
  Densenet_4_4,
  Densenet_4_5,
  Densenet_4_4_DEX
)

def create_fake_input_tensor(image_size):
  return np.zeros([1, image_size, image_size, 3], np.float32)

def test_net(net, x, expected_output_shape):
  with tf.Session() as sess:
    net.init_trainable_weights()
    sess.run(tf.global_variables_initializer())

    res = net.forward(x).eval()
    np.testing.assert_array_equal(expected_output_shape, res.shape)

  tf.reset_default_graph()

class Test_nn(unittest.TestCase):

  def test_Densenet_4_4_FeatureExtractor(self):
    net = Densenet_4_4_FeatureExtractor()
    test_net(net, create_fake_input_tensor(112), [1, 7, 7, 256])

  def test_Densenet_4_5_FeatureExtractor(self):
    net = Densenet_4_5_FeatureExtractor()
    test_net(net, create_fake_input_tensor(112), [1, 7, 7, 512])

  def test_Densenet_4_4(self):
    net = Densenet_4_4()
    test_net(net, create_fake_input_tensor(112), [1])

  def test_Densenet_4_5(self):
    net = Densenet_4_5()
    test_net(net, create_fake_input_tensor(112), [1])

  def test_Densenet_4_4_DEX(self):
    net = Densenet_4_4_DEX()
    test_net(net, create_fake_input_tensor(112), [1])
