from keras.layers import Input

import colabsnippets.face_detection.fpn.keras.layers as L

def make_heads(stage_outputs, stage_anchors, simple_heads=True, is_sigmoid_offsets=False, num_class_outputs=1):
  outputs = []
  for stage_idx, stage_output in enumerate(stage_outputs):
    num_anchors = len(stage_anchors[stage_idx])
    for anchor_idx in range(0, num_anchors):
      # backward compatibility
      if num_anchors == 1:
        anchor_id = 'stage_' + str(stage_idx)
      else:
        anchor_id = 's_' + str(stage_idx) + '_a_' + str(anchor_idx)

      print(anchor_id)

      if simple_heads:
        offsets = L.conv1x1(stage_output, 2,
                            'conv_offsets_' + anchor_id if is_sigmoid_offsets else 'output_' + anchor_id + '_offsets')
        if is_sigmoid_offsets:
          offsets = L.sigmoid(offsets, 'output_' + anchor_id + '_offsets')
        scales = L.conv1x1(stage_output, 2, 'output_' + anchor_id + '_scales')
        scores_logits = L.conv1x1(stage_output, 1, 'conv_scores_' + anchor_id)
      else:
        offsets = L.depthwise_separable_conv2d(stage_output, 2,
                            'conv_offsets_' + anchor_id if is_sigmoid_offsets else 'output_' + anchor_id + '_offsets')
        scales = L.depthwise_separable_conv2d(stage_output, 2, 'output_' + anchor_id + '_scales')
        scores_logits = L.depthwise_separable_conv2d(stage_output, num_class_outputs, 'conv_scores_' + anchor_id)



      scores = L.sigmoid(scores_logits, 'output_' + anchor_id + '_object')
      scores2 = L.sigmoid(scores_logits, 'output_' + anchor_id + '_no_object')

      outputs.append(offsets)
      outputs.append(scales)
      outputs.append(scores)
      outputs.append(scores2)
  return outputs

def create_fpn(bottom_up, detector_channels, stage_anchors, is_sigmoid_offsets=False, stage_strides=None,
               top_down_out_channels=64, input_size=640, with_batch_norm=True, with_top_down=True,
               with_detection_module=True, num_class_outputs=1, simple_heads=True):
  inputs = Input(shape=(input_size, input_size, 3))
  normalized = L.normalize(inputs)

  stage_outputs = bottom_up(normalized)
  print(stage_outputs)

  if with_top_down:
    stage_outputs = [
      L.conv1x1(x, top_down_out_channels, 'conv_top_down_in_channel_shrink_' + str(idx),
                with_batch_norm=with_batch_norm)
      for idx, x in enumerate(stage_outputs)]
    stage_outputs = L.top_down(stage_outputs, [top_down_out_channels for idx in range(0, len(stage_outputs))],
                               'top_down',
                               stage_strides, with_batch_norm=with_batch_norm)
    print(stage_outputs)

  if with_detection_module:
    stage_out_channels = detector_channels if isinstance(detector_channels, list) else [detector_channels for idx in
                                                                                        range(0, len(stage_outputs))]

    print(stage_out_channels)

    stage_outputs = [
      L.ssh_detection_module(out, stage_out_channels[idx], 'det_' + str(idx), with_batch_norm=with_batch_norm) for
      idx, out in enumerate(stage_outputs)]

  return inputs, make_heads(stage_outputs, stage_anchors, simple_heads=simple_heads, num_class_outputs=num_class_outputs, is_sigmoid_offsets=is_sigmoid_offsets)
