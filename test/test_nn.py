import unittest

from colabsnippets.nn import (
  Densenet_4_4_FeatureExtractor,
  Densenet_4_5_FeatureExtractor,
  XceptionTiny
)

from test.utils import create_fake_input_tensor, test_net

class Test_nn(unittest.TestCase):

  def test_Densenet_4_4_FeatureExtractor(self):
    net = Densenet_4_4_FeatureExtractor()
    test_net(net, create_fake_input_tensor(112), [1, 7, 7, 256])

  def test_Densenet_4_5_FeatureExtractor(self):
    net = Densenet_4_5_FeatureExtractor()
    test_net(net, create_fake_input_tensor(112), [1, 7, 7, 512])

  def test_XceptionTiny(self):
    net = XceptionTiny()
    test_net(net, create_fake_input_tensor(112), [1, 7, 7, 512])
