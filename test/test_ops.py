import unittest
import numpy as np
import tensorflow as tf

from colabsnippets.ops import (
  normalize,
  batch_norm,
  conv2d,
  depthwise_separable_conv2d,
  fully_connected,
  dense_block
)


class Test_ops(unittest.TestCase):

  def test_normalize(self):
    with tf.Session() as sess:
      x = np.array([[[[101, 112, 123], [104, 115, 126]]]])
      mean_rgb = [100, 110, 120]
      res = normalize(x, mean_rgb).eval()
      np.testing.assert_array_equal([[[[1 / 256, 2 / 256, 3 / 256], [4 / 256, 5 / 256, 6 / 256]]]], res)
