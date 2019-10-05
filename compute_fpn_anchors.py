import numpy as np

from colabsnippets.face_detection import calculate_iou

# 32 : 1,2,4,8,16 -> 23 - 724px
# 28 : 1,2,4,8,16 -> 20 - 633px
# 16 : 1,2,4,8,16,32 -> 12 - 724px

base_size = 16
anchor_sizes = [base_size * a for a in [1, 2, 4, 8, 16]]
anchors = [[0, 0, a, a] for a in anchor_sizes]

for i in range(0, 860):
  b = [0, 0, i, i]

  ious = np.array([calculate_iou(b, ba) for ba in anchors])
  num_matches = np.where(ious > 0.5)[0].shape[0]
  if num_matches < 1:
    print(i)
