from keras.layers import Lambda, UpSampling2D, Conv2D, SeparableConv2D, MaxPooling2D, \
  BatchNormalization, Activation, Add, Concatenate


def normalize(x, channel_means=123):
  return Lambda(lambda x: (x - channel_means) / 256)(x)


def channel_concat(x, name):
  return Concatenate(axis=3, name=name)(x)


def relu(x, name):
  return Activation('relu', name=name)(x)


def sigmoid(x, name):
  return Activation('sigmoid', name=name)(x)


def max_pooling(x, name):
  return MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name)(x)


def conv(ConvLayer, x, channels_out, name, strides, filter_size, with_batch_norm=False):
  out = x
  out = ConvLayer(channels_out, filter_size, strides=strides, padding='same', use_bias=not with_batch_norm, name=name)(
    out)
  if with_batch_norm:
    out = BatchNormalization(name=name + "/batch_norm")(out)
  return out


def conv1x1(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (1, 1), (1, 1), with_batch_norm=with_batch_norm)


def conv1x1_down(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (2, 2), (1, 1), with_batch_norm=with_batch_norm)


def conv2d(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (1, 1), (3, 3), with_batch_norm=with_batch_norm)


def conv2d_down(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (2, 2), (3, 3), with_batch_norm=with_batch_norm)


def depthwise_separable_conv2d(x, channels_out, name, with_batch_norm=False):
  return conv(SeparableConv2D, x, channels_out, name, (1, 1), (3, 3), with_batch_norm=with_batch_norm)


def add(name, x, y):
  return Add(name=name)([x, y])


def depthwise_separable_conv2d(x, channels_out, name, strides=(1, 1), with_batch_norm=False):
  return conv(SeparableConv2D, x, channels_out, name, strides, (3, 3), with_batch_norm=with_batch_norm)


def depthwise_separable_conv2d_down(x, channels_out, name, with_batch_norm=False):
  return depthwise_separable_conv2d(x, channels_out, name, strides=(2, 2), with_batch_norm=with_batch_norm)


def xception_reduction_module(x, channels_out, name, activate_inputs=True, with_batch_norm=False):
  out = x
  out = relu(out, name + "/relu_input") if activate_inputs else out
  out = depthwise_separable_conv2d(out, channels_out, name + '/separable_conv0', with_batch_norm=with_batch_norm)
  out = relu(out, name + '/relu_intermediate')
  out = depthwise_separable_conv2d(out, channels_out, name + '/separable_conv1', with_batch_norm=with_batch_norm)
  out = max_pooling(out, name + "/max_pooling")
  expand = conv1x1_down(x, channels_out, name + '/expansion_conv', with_batch_norm=with_batch_norm)
  return Add(name=name + '/skip_connection')([out, expand])


def xception_main_module(x, channels, name, num_convs=3, with_batch_norm=False, is_channel_expand=False):
  skip0 = x
  out = x
  for i in range(0, num_convs):
    out = depthwise_separable_conv2d(relu(out, name + '/relu' + str(i)), channels, name + '/separable_conv' + str(i),
                                     with_batch_norm=with_batch_norm)
    if is_channel_expand and i == 0:
      skip0 = out
  return Add(name=name + '/skip_connection')([out, skip0])


def ssh_context_module(x, channels, name, with_batch_norm=False):
  shrink = out = relu(conv2d(x, channels, name + '/conv_shrink', with_batch_norm=with_batch_norm),
                      name + '/conv_shrink/relu')
  out = relu(conv2d(out, channels, name + '/conv_1', with_batch_norm=with_batch_norm), name + '/conv_1/relu')
  out1 = conv2d(out, channels, name + '/conv_out_1', with_batch_norm=with_batch_norm)
  out2 = conv2d(shrink, channels, name + '/conv_out_2', with_batch_norm=with_batch_norm)
  return out1, out2


def ssh_detection_module(x, out_channels, name, with_batch_norm=False):
  ctx_out_1, ctx_out_2 = ssh_context_module(x, int(out_channels / 4), name + '/ctx', with_batch_norm=with_batch_norm)
  shrink = conv2d(x, int(out_channels / 2), name + '/conv_shrink', with_batch_norm=with_batch_norm)
  out = channel_concat([shrink, ctx_out_1, ctx_out_2], name + '/channel_concat')
  return relu(out, name + '/relu')


def top_down(stage_outputs, stage_out_channels, name, stage_strides, with_batch_norm=False):
  up = stage_outputs[0]
  for idx in range(1, len(stage_outputs)):
    if stage_strides is None or stage_strides[idx - 1] != stage_strides[idx]:
      up = UpSampling2D((2, 2), name=name + '/upsample_' + str(idx))(up)
    print(up)
    up = x = Add(name=name + '/add_' + str(idx))([up, stage_outputs[idx]])
    x = depthwise_separable_conv2d(x, stage_out_channels[idx], name + '/conv_anti_aliasing_' + str(idx),
                                   with_batch_norm=with_batch_norm)
    stage_outputs[idx] = relu(x, name + '/relu_' + str(idx))

  return stage_outputs
