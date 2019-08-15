import unittest

from colabsnippets.face_detection import (
  calculate_iou
)

class Test_calculate_iou(unittest.TestCase):

  def test_absolute_box(self):
    box1 = (10, 10, 20, 20)
    self.assertEqual(1.0, calculate_iou(box1, box1))
    self.assertEqual(0.5, calculate_iou(box1, (10, 10, 10, 20)))
    self.assertEqual(0.25, calculate_iou(box1, (10, 10, 40, 40)))
    self.assertEqual(0, calculate_iou(box1, (0, 0, 10, 10)))
    self.assertEqual(0, calculate_iou(box1, (5, 5, 2.5, 2.5)))
    self.assertEqual(0, calculate_iou(box1, (-10, -10, 10, 10)))
    self.assertEqual(0, calculate_iou(box1, (0, 40, 40, 40)))

  def test_relative_box(self):
    box1 = (0.1, 0.1, 0.2, 0.2)
    self.assertEqual(1.0, round(calculate_iou(box1, box1), 8))
    self.assertEqual(0.5, round(calculate_iou(box1, (0.1, 0.1, 0.1, 0.2)), 8))
    self.assertEqual(0.25, round(calculate_iou(box1, (0.1, 0.1, 0.4, 0.4)), 8))
    self.assertEqual(0, calculate_iou(box1, (0, 0, 0.1, 0.1)))
    self.assertEqual(0, calculate_iou(box1, (0.05, 0.05, 0.025, 0.025)))
    self.assertEqual(0, calculate_iou(box1, (-0.1, -0.1, 0.1, 0.1)))
    self.assertEqual(0, calculate_iou(box1, (0, 0.4, 0.4, 0.4)))

