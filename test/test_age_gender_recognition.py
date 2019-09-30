import unittest

from colabsnippets.age_gender_recognition import (
  AgeGenderXceptionTiny
)

from test.utils import create_fake_input_tensor, test_net

class Test_age_recognition(unittest.TestCase):

  def test_AgeGenderXceptionTiny(self):
    net = AgeGenderXceptionTiny()
    test_net(net, create_fake_input_tensor(112), ([1], [1, 2]))
