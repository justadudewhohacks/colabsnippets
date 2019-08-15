def create_anchor(w, aspect_ratio = 1.5):
  return (w, aspect_ratio * w)


def generate_anchors(min_anchor_size = 16, max_anchor_size = 416, num_anchors_per_stage = 3, num_stages = 5):
  size_range = max_anchor_size - min_anchor_size
  step = size_range / (num_anchors_per_stage * num_stages)
  anchors_by_stage = []
  for s in range(0, num_stages):
    anchors = []
    for a in range(0, num_anchors_per_stage):
      anchor_idx = s * num_anchors_per_stage + a
      anchors.append(create_anchor(step*anchor_idx + min_anchor_size))

    anchors_by_stage.append(anchors)

  return anchors_by_stage