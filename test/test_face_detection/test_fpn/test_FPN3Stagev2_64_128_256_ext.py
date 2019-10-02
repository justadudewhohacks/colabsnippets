import unittest

from colabsnippets.face_detection.fpn.FPN3Stagev2_64_128_256_ext import FPN3Stagev2_64_128_256_ext
from test.utils import create_fake_input_tensor, test_net_save_load_forward


class Test_FPN3Stagev2_64_128_256_ext(unittest.TestCase):
  def test_FPN3Stagev2_64_128_256_ext(self):
    net = FPN3Stagev2_64_128_256_ext()
    self.assertEquals(3, len(net.anchors))
    self.assertEquals(2, len(net.anchors[0]))
    self.assertEquals(1, net.stage_idx_offset)

  def test_save_load_forward(self):
    net = FPN3Stagev2_64_128_256_ext()
    forward = lambda: net.forward(create_fake_input_tensor(224), 1, 224)
    test_net_save_load_forward(net, forward)

  def test_save_load_forward_no_bn(self):
    net = FPN3Stagev2_64_128_256_ext(with_batch_norm=False)
    forward = lambda: net.forward(create_fake_input_tensor(224), 1, 224)
    test_net_save_load_forward(net, forward)
