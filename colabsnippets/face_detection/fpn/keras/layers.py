from keras.layers import Lambda, Conv2D, SeparableConv2D, MaxPooling2D, \
  BatchNormalization, Activation, Add, Concatenate, DepthwiseConv2D, UpSampling2D


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


class Convolution:
  def __init__(self, ConvType, channels_out, name, strides, filter_size, with_batch_norm=False):
    self.conv = ConvType(channels_out, filter_size, strides=strides, padding='same', use_bias=not with_batch_norm,
                         name=name)
    self.bn_or_identity = BatchNormalization(name=name + "/batch_norm") if with_batch_norm else lambda x: x

  def __call__(self, x):
    return self.bn_or_identity(self.conv(x))


class DepthwiseSeparableConvolution:
  def __init__(self, channels_out, name, strides=(1, 1), filter_size=(3, 3), with_batch_norm=False):
    self.conv = Convolution(SeparableConv2D, channels_out, name, strides, filter_size, with_batch_norm=with_batch_norm)

  def __call__(self, x):
    return self.conv(x)


class Conv1x1:
  def __init__(self, channels_out, name, strides=(1, 1), with_batch_norm=False):
    self.conv = Convolution(Conv2D, channels_out, name, strides, (1, 1), with_batch_norm=with_batch_norm)

  def __call__(self, x):
    return self.conv(x)


class XceptionReductionModule:
  def __init__(self, channels_out, name, activate_inputs=True, with_batch_norm=False):
    self.relu_in_or_identity = Activation('relu', name=name + "/relu_input") if activate_inputs else lambda x: x
    self.dws_conv0 = DepthwiseSeparableConvolution(channels_out, name + '/separable_conv0',
                                                   with_batch_norm=with_batch_norm)
    self.relu_intermediate = Activation('relu', name=name + "/relu_intermediate")
    self.dws_conv1 = DepthwiseSeparableConvolution(channels_out, name + '/separable_conv1',
                                                   with_batch_norm=with_batch_norm)
    self.max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=name + "/max_pooling")
    self.conv_down = Conv1x1(channels_out, name + '/expansion_conv', strides=(2, 2), with_batch_norm=with_batch_norm)
    self.skip = Add(name=name + '/skip_connection')

  def __call__(self, x):
    return self.skip([
      self.max_pool(self.dws_conv1(self.relu_intermediate(self.dws_conv0(self.relu_in_or_identity(x))))),
      self.conv_down(x)
    ])


class XceptionMainModule:
  def __init__(self, channels, name, num_convs=3, with_batch_norm=False):
    self.relus = [Activation('relu', name=name + "/relu" + str(i)) for i in range(0, num_convs)]
    self.convs = [
      DepthwiseSeparableConvolution(channels, name + '/separable_conv' + str(i), with_batch_norm=with_batch_norm) for i
      in range(0, num_convs)]
    self.skip = Add(name=name + '/skip_connection')

  def __call__(self, x, is_channel_expand=False):
    skip0 = out = x
    for i, (relu, conv) in enumerate(zip(self.relus, self.convs)):
      out = conv(relu(out))
      if is_channel_expand and i == 0:
        skip0 = out
    return self.skip([out, skip0])


def conv(ConvType, x, channels_out, name, strides, filter_size, with_batch_norm=False):
  return Convolution(ConvType, channels_out, name, strides, filter_size, with_batch_norm=with_batch_norm)(x)


def conv1x1(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (1, 1), (1, 1), with_batch_norm=with_batch_norm)


def conv1x1_down(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (2, 2), (1, 1), with_batch_norm=with_batch_norm)


def conv2d(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (1, 1), (3, 3), with_batch_norm=with_batch_norm)


def conv2d_down(x, channels_out, name, with_batch_norm=False):
  return conv(Conv2D, x, channels_out, name, (2, 2), (3, 3), with_batch_norm=with_batch_norm)


def depthwise_separable_conv2d(x, channels_out, name, strides=(1, 1), with_batch_norm=False):
  return DepthwiseSeparableConvolution(channels_out, name, strides, filter_size=(3, 3),
                                       with_batch_norm=with_batch_norm)(x)


def depthwise_separable_conv2d_down(x, channels_out, name, with_batch_norm=False):
  return depthwise_separable_conv2d(x, channels_out, name, strides=(2, 2), with_batch_norm=with_batch_norm)


def depthwise_conv2d(x, name, strides=(1, 1), with_batch_norm=False):
  out = DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=not with_batch_norm,
                        name=name + "/depthwise_conv")(x)
  if with_batch_norm:
    out = BatchNormalization(name=name + "/batch_norm")(out)
  return out


def depthwise_separable_conv2d_with_intermediate(x, channels_out, name, strides=(1, 1), with_batch_norm=False):
  out = depthwise_conv2d(x, name + "/depthwise_conv", strides=strides, with_batch_norm=with_batch_norm)
  out = relu(out, name + '/relu_intermediate')
  out = conv1x1(out, channels_out, name=name + "/pointwise_conv", with_batch_norm=with_batch_norm)
  return out


def depthwise_separable_conv2d_down_with_intermediate(x, channels_out, name, with_batch_norm=False):
  return depthwise_separable_conv2d_with_intermediate(x, channels_out, name, strides=(2, 2), with_batch_norm=with_batch_norm)


def xception_reduction_module(x, channels_out, name, activate_inputs=True, with_batch_norm=False):
  return XceptionReductionModule(channels_out, name, activate_inputs=activate_inputs, with_batch_norm=with_batch_norm)(
    x)


def xception_main_module(x, channels, name, num_convs=3, with_batch_norm=False, is_channel_expand=False):
  return XceptionMainModule(channels, name, num_convs=num_convs, with_batch_norm=with_batch_norm)(x,
                                                                                                  is_channel_expand=is_channel_expand)


def mnetv2_conv(x, channels_in, exp_factor, name, is_conv_down=False, channels_out=None, with_batch_norm=False):
  channels_out = channels_out if channels_out is not None else channels_in
  out = relu(x, name + "/relu_input")
  out = conv1x1(out, int(channels_in * exp_factor), name + '/expansion_conv', with_batch_norm=with_batch_norm)
  out = relu(out, name + "/relu_expansion_conv")
  out = depthwise_separable_conv2d(out, channels_out, name + '/separable_conv',
                                   strides=(2, 2) if is_conv_down else (1, 1), with_batch_norm=with_batch_norm)
  out = out if is_conv_down else Add(name=name + '/skip_connection')([x, out])
  return out


def ssh_context_module(x, channels, name, with_batch_norm=False):
  shrink = out = relu(conv2d(x, channels, name + '/conv_shrink', with_batch_norm=with_batch_norm),
                      name + '/conv_shrink/relu')
  out = relu(conv2d(out, channels, name + '/conv_1', with_batch_norm=with_batch_norm), name + '/conv_1/relu')
  out1 = conv2d(out, channels, name + '/conv_out_1', with_batch_norm=with_batch_norm)
  out2 = conv2d(shrink, channels, name + '/conv_out_2', with_batch_norm=with_batch_norm)
  return out1, out2


def ssh_detection_module(x, out_channels, name, with_batch_norm=False):
  if out_channels == 0:
    return x
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
