import unittest

from colabsnippets.nn import (
  Densenet_4_4_FeatureExtractor,
  Densenet_4_5_FeatureExtractor,
  Densenet_4_4,
  Densenet_4_5,
  Densenet_4_4_DEX
)

class Test_nn(unittest.TestCase):

  def test_Densenet_4_4_FeatureExtractor(self):
    net = Densenet_4_4_FeatureExtractor()

  def test_Densenet_4_5_FeatureExtractor(self):
    net = Densenet_4_5_FeatureExtractor()

  def test_Densenet_4_4(self):
    net = Densenet_4_4()

  def test_Densenet_4_5(self):
    net = Densenet_4_5()

  def test_Densenet_4_4_DEX(self):
    net = Densenet_4_4_DEX()
