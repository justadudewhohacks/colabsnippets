import unittest

from colabsnippets.face_detection.fpn.FPN3Stagev3_64_128_256 import FPN3Stagev3_64_128_256
from test.utils import create_fake_input_tensor, test_net_save_load_forward


class Test_FPN3Stagev3_64_128_256(unittest.TestCase):
  def test_FPN3Stagev3_64_128_256(self):
    net = FPN3Stagev3_64_128_256()
    self.assertEquals(3, len(net.anchors))
    self.assertEquals(2, len(net.anchors[0]))
    self.assertEquals(1, net.stage_idx_offset)

  def test_save_load_forward(self):
    net = FPN3Stagev3_64_128_256()
    forward = lambda: net.forward(create_fake_input_tensor(224), 1, 224)
    test_net_save_load_forward(net, forward)
