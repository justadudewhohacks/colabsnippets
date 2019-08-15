import unittest

from colabsnippets.face_detection.fpn import (
  FPNBase
)

class Test_FPNBase(unittest.TestCase):

  def test_FPNBase(self):
    net = FPNBase()
    self.assertEquals(5, len(net.anchors))
    self.assertEquals(3, len(net.anchors[0]))
    self.assertEquals(0, net.stage_idx_offset)

