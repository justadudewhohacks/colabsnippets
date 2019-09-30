import unittest

from colabsnippets.age_recognition import (
  Densenet_4_4,
  Densenet_4_5,
  Densenet_4_4_DEX,
  MobilenetV2,
  AgeXceptionTiny
)

from test.utils import create_fake_input_tensor, test_net

class Test_age_recognition(unittest.TestCase):

  def test_Densenet_4_4(self):
    net = Densenet_4_4()
    test_net(net, create_fake_input_tensor(112), [1])

  def test_Densenet_4_5(self):
    net = Densenet_4_5()
    test_net(net, create_fake_input_tensor(112), [1])

  def test_Densenet_4_4_DEX(self):
    net = Densenet_4_4_DEX()
    test_net(net, create_fake_input_tensor(112), [1])

  def test_MobilenetV2(self):
    net = MobilenetV2()
    test_net(net, create_fake_input_tensor(112), [1])

  def test_AgeXceptionTiny(self):
    net = AgeXceptionTiny()
    test_net(net, create_fake_input_tensor(112), [1])
