import tensorflow as tf
import numpy as np

from .Densenet_4_4_FeatureExtractor import Densenet_4_4_FeatureExtractor
from ..ops import fully_connected

class AgeCategoryClassifier(Densenet_4_4_FeatureExtractor):
  def __init__(self, name = 'age_category_classifier', age_categories = [8, 18, 30, 45, 60, 120], channel_multiplier = 1.0):
    super().__init__(name = name, channel_multiplier = channel_multiplier)
    self.age_categories = np.asarray(age_categories)

  def get_age_category_one_hot_vectors(self, ages):
    one_hots = []
    for age in ages:
      one_hots.append(self.get_age_category_one_hot_vector(age))
    return one_hots

  def get_age_category_one_hot_vector(self, age):
    if age < 0 or age > self.age_categories[-1]:
      raise Exception('get_age_one_hot_vector - invalid age: ' + str(age))
    category_idx = max(0, np.where(self.age_categories >= age)[0][0])
    one_hot = np.zeros(len(self.age_categories))
    one_hot[category_idx] = 1
    return one_hot

  def initialize_weights(self, weight_processor):
    super().initialize_weights(weight_processor)
    with tf.variable_scope(self.name):
      weight_processor.process_fc_weights(int(self.channel_multiplier * 256), len(self.age_categories), 'fc_age_category')

  def forward(self, batch_tensor):
    out = super().forward(batch_tensor)
    with tf.variable_scope(self.name, reuse = True):
      out = tf.nn.avg_pool(out, [1, 7, 7, 1], [1, 2, 2, 1], 'VALID')
      out = fully_connected(out, 'fc_age_category')
    return out