import unittest

from colabsnippets.face_detection.object_detector import (
  ObjectDetectorBase
)

class Test_ObjectDetectorBase(unittest.TestCase):

  def test_ObjectDetectorBase(self):
    net = ObjectDetectorBase(num_cells = 20)
    self.assertEquals(20, net.num_cells)

