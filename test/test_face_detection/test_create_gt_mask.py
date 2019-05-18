import unittest
import numpy as np

from colabsnippets.face_detection import (
  create_gt_mask
)

class Test_create_gt_mask(unittest.TestCase):

  def test_create_gt_mask_single_anchor(self):
    anchors = [(1.0, 1.0)]
    boxes = [
      (0, 0, 0.1, 0.1), (0.1, 0, 0.1, 0.1), (0, 0.1, 0.1, 0.1), (0.4, 0.5, 0.1, 0.1)
    ]

    mask = create_gt_mask([boxes], 10, anchors)
    expected_shape = [1, 10, 10, 1, 1]
    expected_mask = np.zeros(expected_shape)
    expected_mask[0, 0, 0, 0, 0] = 1
    expected_mask[0, 1, 0, 0, 0] = 1
    expected_mask[0, 0, 1, 0, 0] = 1
    expected_mask[0, 4, 5, 0, 0] = 1

    np.testing.assert_array_equal(expected_shape, mask.shape)
    np.testing.assert_array_equal(expected_mask, mask)

  def test_create_gt_mask_multiple_anchors(self):
    anchors = [(1.0, 1.0), (0.9, 0.8), (0.8, 0.9)]
    boxes = [
      (0, 0, 0.1, 0.1), (0.1, 0, 0.1, 0.1), (0, 0.1, 0.1, 0.1),
      (0, 0, 0.09, 0.08), (0.2, 0, 0.09, 0.08), (0, 0.2, 0.09, 0.08),
      (0, 0, 0.08, 0.09), (0.3, 0, 0.08, 0.09), (0, 0.3, 0.08, 0.09)
    ]

    mask = create_gt_mask([boxes], 10, anchors)

    expected_shape = [1, 10, 10, 3, 1]
    expected_mask = np.zeros(expected_shape)
    expected_mask[0, 0, 0, 0, 0] = 1
    expected_mask[0, 1, 0, 0, 0] = 1
    expected_mask[0, 0, 1, 0, 0] = 1
    expected_mask[0, 0, 0, 1, 0] = 1
    expected_mask[0, 2, 0, 1, 0] = 1
    expected_mask[0, 0, 2, 1, 0] = 1
    expected_mask[0, 0, 0, 2, 0] = 1
    expected_mask[0, 3, 0, 2, 0] = 1
    expected_mask[0, 0, 3, 2, 0] = 1

    np.testing.assert_array_equal(expected_shape, mask.shape)
    np.testing.assert_array_equal(expected_mask, mask)
