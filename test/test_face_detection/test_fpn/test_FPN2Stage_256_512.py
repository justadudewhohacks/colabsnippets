import unittest

from colabsnippets.face_detection.fpn.FPN2Stage_256_512 import FPN2Stage_256_512
from test.utils import create_fake_input_tensor, test_net_save_load_forward


class Test_FPN2Stage_256_512(unittest.TestCase):

  def test_FPN2Stage_256_512(self):
    net = FPN2Stage_256_512()
    self.assertEquals(2, len(net.anchors))
    self.assertEquals(3, len(net.anchors[0]))
    self.assertEquals(2, len(net.anchors[1]))
    self.assertEquals(3, net.stage_idx_offset)

  def test_save_load_forward(self):
    net = FPN2Stage_256_512()
    forward = lambda: net.forward(create_fake_input_tensor(224), 1, 224)
    test_net_save_load_forward(net, forward)
