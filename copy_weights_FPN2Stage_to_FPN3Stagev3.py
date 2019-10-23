import numpy as np
import tensorflow as tf

from colabsnippets.face_detection.fpn.FPN2Stage_256_512 import FPN2Stage_256_512
from colabsnippets.face_detection.fpn.FPN2Stage_64_128 import FPN2Stage_64_128
from colabsnippets.face_detection.fpn.FPN3Stagev3_128_256_512 import FPN3Stagev3_128_256_512
from colabsnippets.face_detection.fpn.FPN3Stagev3_32_64_128 import FPN3Stagev3_32_64_128

tf.reset_default_graph()

net_from = FPN2Stage_64_128()
net_to = FPN3Stagev3_32_64_128()
src_checkpoint = './fpn2stage_64_128_no_wider_224_1216_32_epoch743'

# net_from = FPN2Stage_128_256()
# net_to = FPN3Stagev3_64_128_256()
# src_checkpoint = './fpn2stage_128_256_no_wider_crop_224_1216_32_epoch664'

#net_from = FPN2Stage_256_512()
#net_to = FPN3Stagev3_128_256_512()
#src_checkpoint = './fpn2stage_256_512_no_wider_224_1216_32_epoch492'

net_from.load_weights(src_checkpoint, net_json_file=net_from.name + '.json')
net_to.init_trainable_weights()
net_to.save_meta_json(net_to.name)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  net_to.save_weights('ref')
  print([(var.name.replace(net_from.name, net_to.name), var.shape) for var in
         net_from.get_net_vars_in_initialization_order()])
  print([(var.name, var.shape) for var in net_to.get_net_vars_in_initialization_order()])

  var_from_map = {}
  for var in net_from.get_net_vars_in_initialization_order():
    name = var.name.replace(net_from.name, net_to.name)
    if name.startswith(net_to.name + '/top_down/conv_shrink_1/'):
      name = name.replace(net_to.name + '/top_down/conv_shrink_1/',
                          net_to.name + '/top_down/conv_shrink_2/')

    elif name.startswith(net_to.name + '/top_down/conv_shrink_2/'):
      name = name.replace(net_to.name + '/top_down/conv_shrink_2/',
                          net_to.name + '/top_down/conv_shrink_3/')

    elif name.startswith(net_to.name + '/top_down/conv_anti_aliasing_1/'):
      name = name.replace(net_to.name + '/top_down/conv_anti_aliasing_1/',
                          net_to.name + '/top_down/conv_anti_aliasing_2/')

    elif name.startswith(net_to.name + '/top_down/conv_anti_aliasing_2/'):
      name = name.replace(net_to.name + '/top_down/conv_anti_aliasing_2/',
                          net_to.name + '/top_down/conv_anti_aliasing_3/')

    elif name.startswith(net_to.name + '/det_1/'):
      name = name.replace(net_to.name + '/det_1/',
                          net_to.name + '/det_2/')

    elif name.startswith(net_to.name + '/det_2/'):
      name = name.replace(net_to.name + '/det_2/',
                          net_to.name + '/det_3/')

    elif name.startswith(net_to.name + '/classifier/conv_out_0/'):
      name = name.replace(net_to.name + '/classifier/conv_out_0/',
                          net_to.name + '/classifier/conv_out_1/')
    elif name.startswith(net_to.name + '/classifier/conv_out_1/'):
      name = name.replace(net_to.name + '/classifier/conv_out_1/',
                          net_to.name + '/classifier/conv_out_2/')

    var_from_map[name] = var

  old_classifier0_filter = var_from_map[net_to.name + "/classifier/conv_out_1/filter:0"]
  old_classifier0_bias = var_from_map[net_to.name + "/classifier/conv_out_1/bias:0"]

  #print(old_classifier0_filter)
  #print(old_classifier0_bias)
  out_data = np.array([], dtype='float32')
  for var in net_to.get_net_vars_in_initialization_order():
    #print(var)
    if net_to.name + "/classifier/conv_out_0/filter:0" == var.name:
      front = tf.slice(var, [0, 0, 0, 0], [1, 1, net_to.out_channels, 5])
      back = tf.slice(old_classifier0_filter, [0, 0, 0, 0], [1, 1, net_to.out_channels, 5])
      classifier0_filter = tf.concat([front, back], axis=3)
      out_data = np.append(out_data, classifier0_filter.eval())
      continue
    if net_to.name + "/classifier/conv_out_0/bias:0" == var.name:
      front = tf.slice(var, [0], [5])
      back = tf.slice(old_classifier0_bias, [0], [5])
      #print(front)
      #print(back)
      classifier0_bias = tf.concat([front, back], axis=0)
      out_data = np.append(out_data, classifier0_bias.eval())
      continue
    if net_to.name + "/classifier/conv_out_1/filter:0" == var.name:
      classifier1_filter = tf.slice(old_classifier0_filter, [0, 0, 0, 5], [1, 1, net_to.out_channels, 10])
      out_data = np.append(out_data, classifier1_filter.eval())
      continue
    if net_to.name + "/classifier/conv_out_1/bias:0" == var.name:
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
