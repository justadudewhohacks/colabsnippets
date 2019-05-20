import unittest

from colabsnippets.face_detection.yolov2 import (
  get_box_grid_position
)

class Test_get_box_grid_position(unittest.TestCase):

  def test_grid_position_0_0(self):
    col, row, anchor_idx = get_box_grid_position((0, 0, 0.1, 0.1), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(0, col)
    self.assertEqual(0, row)

  def test_grid_position_5_5(self):
    col, row, anchor_idx = get_box_grid_position((0.5, 0.5, 0.1, 0.1), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(5, col)
    self.assertEqual(5, row)

  def test_grid_position_2_4(self):
    col, row, anchor_idx = get_box_grid_position((0.2, 0.4, 0.1, 0.1), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(2, col)
    self.assertEqual(4, row)

  def test_grid_position_9_9(self):
    col, row, anchor_idx = get_box_grid_position((1.0, 1.0, 0.1, 0.1), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(9, col)
    self.assertEqual(9, row)


  def test_grid_position_ct_1_1(self):
    col, row, anchor_idx = get_box_grid_position((0, 0, 0.2, 0.2), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(1, col)
    self.assertEqual(1, row)

  def test_grid_position_ct_6_6(self):
    col, row, anchor_idx = get_box_grid_position((0.5, 0.5, 0.2, 0.2), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(6, col)
    self.assertEqual(6, row)

  def test_grid_position_ct_3_5(self):
    col, row, anchor_idx = get_box_grid_position((0.2, 0.4, 0.2, 0.2), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(3, col)
    self.assertEqual(5, row)

  def test_grid_position_exceeds_negative_0_0(self):
    col, row, anchor_idx = get_box_grid_position((-0.5, -0.5, 0.1, 0.1), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(0, col)
    self.assertEqual(0, row)

  def test_grid_position_exceeds_positive_9_9(self):
    col, row, anchor_idx = get_box_grid_position((1.5, 1.5, 0.1, 0.1), 10, [(1.0, 1.0)])
    self.assertEqual(0, anchor_idx)
    self.assertEqual(9, col)
    self.assertEqual(9, row)

  def test_anchor_match(self):
    anchors = [(2.0, 2.0), (1.0, 1.0), (1.5, 1.0)]
    _, __, anchor_idx1 = get_box_grid_position((0, 0, 0.1, 0.1), 10, anchors)
    _, __, anchor_idx2 = get_box_grid_position((0, 0, 0.2, 0.2), 10, anchors)
    _, __, anchor_idx3 = get_box_grid_position((0, 0, 0.15, 0.1), 10, anchors)
    self.assertEqual(1, anchor_idx1)
    self.assertEqual(0, anchor_idx2)
    self.assertEqual(2, anchor_idx3)