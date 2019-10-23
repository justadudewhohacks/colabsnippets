import numpy as np
import tensorflow as tf

from colabsnippets.face_detection.fpn.FPN2Stage_128_256 import FPN2Stage_128_256
from colabsnippets.face_detection.fpn.FPN3Stagev3_64_128_256 import FPN3Stagev3_64_128_256

tf.reset_default_graph()

net_from = FPN2Stage_128_256()
net_to = FPN3Stagev3_64_128_256()

net_from.load_weights('./fpn2stage_128_256_no_wider_crop_224_1216_32_epoch664', net_json_file=net_from.name + '.json')
net_to.init_trainable_weights()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print([(var.name.replace('fpn2stage_128_256', 'fpn3stagev3_64_128_256'), var.shape) for var in
         net_from.get_net_vars_in_initialization_order()])
  print([(var.name, var.shape) for var in net_to.get_net_vars_in_initialization_order()])

  var_from_map = {}
  for var in net_from.get_net_vars_in_initialization_order():
    name = var.name.replace('fpn2stage_128_256', 'fpn3stagev3_64_128_256')
    if name.startswith('fpn3stagev3_64_128_256/top_down/conv_shrink_1/'):
      name = name.replace('fpn3stagev3_64_128_256/top_down/conv_shrink_1/',
                          'fpn3stagev3_64_128_256/top_down/conv_shrink_2/')

    elif name.startswith('fpn3stagev3_64_128_256/top_down/conv_shrink_2/'):
      name = name.replace('fpn3stagev3_64_128_256/top_down/conv_shrink_2/',
                          'fpn3stagev3_64_128_256/top_down/conv_shrink_3/')

    elif name.startswith('fpn3stagev3_64_128_256/top_down/conv_anti_aliasing_1/'):
      name = name.replace('fpn3stagev3_64_128_256/top_down/conv_anti_aliasing_1/',
                          'fpn3stagev3_64_128_256/top_down/conv_anti_aliasing_2/')

    elif name.startswith('fpn3stagev3_64_128_256/top_down/conv_anti_aliasing_2/'):
      name = name.replace('fpn3stagev3_64_128_256/top_down/conv_anti_aliasing_2/',
                          'fpn3stagev3_64_128_256/top_down/conv_anti_aliasing_3/')

    elif name.startswith('fpn3stagev3_64_128_256/det_1/'):
      name = name.replace('fpn3stagev3_64_128_256/det_1/',
                          'fpn3stagev3_64_128_256/det_2/')

    elif name.startswith('fpn3stagev3_64_128_256/det_2/'):
      name = name.replace('fpn3stagev3_64_128_256/det_2/',
                          'fpn3stagev3_64_128_256/det_3/')

    elif name.startswith('fpn3stagev3_64_128_256/classifier/conv_out_0/'):
      name = name.replace('fpn3stagev3_64_128_256/classifier/conv_out_0/',
                          'fpn3stagev3_64_128_256/classifier/conv_out_1/')
    elif name.startswith('fpn3stagev3_64_128_256/classifier/conv_out_1/'):
      name = name.replace('fpn3stagev3_64_128_256/classifier/conv_out_1/',
                          'fpn3stagev3_64_128_256/classifier/conv_out_2/')

    var_from_map[name] = var

  old_classifier0_filter = var_from_map["fpn3stagev3_64_128_256/classifier/conv_out_1/filter:0"]
  old_classifier0_bias = var_from_map["fpn3stagev3_64_128_256/classifier/conv_out_1/bias:0"]

  print(old_classifier0_filter)
  print(old_classifier0_bias)
  out_data = np.array([], dtype='float32')
  for var in net_to.get_net_vars_in_initialization_order():
    print(var)
    if "fpn3stagev3_64_128_256/classifier/conv_out_0/filter:0" == var.name:
      front = tf.slice(var, [0, 0, 0, 0], [1, 1, 128, 5])
      back = tf.slice(old_classifier0_filter, [0, 0, 0, 0], [1, 1, 128, 5])
      classifier0_filter = tf.concat([front, back], axis=3)
      out_data = np.append(out_data, classifier0_filter.eval())
      continue
    if "fpn3stagev3_64_128_256/classifier/conv_out_0/bias:0" == var.name:
      front = tf.slice(var, [0], [5])
      back = tf.slice(old_classifier0_bias, [0], [5])
      print(front)
      print(back)
      classifier0_bias = tf.concat([front, back], axis=0)
      out_data = np.append(out_data, classifier0_bias.eval())
      continue
    if "fpn3stagev3_64_128_256/classifier/conv_out_1/filter:0" == var.name:
      classifier1_filter = tf.slice(old_classifier0_filter, [0, 0, 0, 5], [1, 1, 128, 10])
      out_data = np.append(out_data, classifier1_filter.eval())
      continue
    if "fpn3stagev3_64_128_256/classifier/conv_out_1/bias:0" == var.name:
      classifier1_bias = tf.slice(old_classifier0_bias, [5], [10])
      out_data = np.append(out_data, classifier1_bias.eval())
      continue

    if not var.name in var_from_map:
      print('not in source:', var.name, var.shape)
      out_data = np.append(out_data, var.eval().flatten())
    elif not var.shape == var_from_map[var.name].shape:
      print('shape not matching:', var.name, var.shape, var.shape, var_from_map[var.name].shape)
      raise Exception('fatal')
    else:
      out_data = np.append(out_data, var_from_map[var.name].eval().flatten())

  np.save(net_to.name + '_epoch0', out_data)


tf.reset_default_graph()
net_to.load_weights(net_to.name + '_epoch0', net_json_file=net_to.name + '.json')
