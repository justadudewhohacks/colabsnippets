import unittest
import numpy as np
import math

from colabsnippets.face_detection import (
  create_gt_coords,
  inverse_sigmoid
)

class Test_create_gt_coords(unittest.TestCase):

  def test_create_gt_coords_single_anchor(self):
    anchors = [(1.0, 1.0)]
    boxes = [
      (0, 0, 0.1, 0.1),
      #(0, 1.1, 0.05, 0.2),
      (0.11, 0.11, 0.1, 0.1),
      (0.43, 0.52, 0.09, 0.12)
    ]

    gt_coords = create_gt_coords([boxes], 10, anchors)
    expected_shape = [1, 10, 10, 1, 4]
    expected_gt_coords = np.zeros(expected_shape)
    expected_gt_coords[0, 0, 0, 0, :] = [inverse_sigmoid(0.5), inverse_sigmoid(0.5), math.log(1), math.log(1)]
    #expected_gt_coords[0, 0, 9, 0, :] = [0, 0, math.log(0.5), math.log(2)]
    expected_gt_coords[0, 1, 1, 0, :] = [inverse_sigmoid(0.6), inverse_sigmoid(0.6), math.log(1), math.log(1)]
    expected_gt_coords[0, 4, 5, 0, :] = [
      inverse_sigmoid(0.3 + (0.09 * 10 / 2)),
      inverse_sigmoid(0.2 + (0.12 * 10 / 2)),
      math.log(0.9),
      math.log(1.2)
    ]

    np.testing.assert_array_almost_equal(expected_shape, gt_coords.shape)
    np.testing.assert_array_almost_equal(expected_gt_coords, gt_coords)
