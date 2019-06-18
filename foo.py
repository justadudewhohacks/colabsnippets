import numpy as np
import math

from colabsnippets.face_detection.yolov2 import (
  create_gt_mask,
  create_gt_coords,
  reconstruct_box
)

anchors = [
  [6.1085124, 4.158409 ],
  [6.4581966, 8.347602 ],
  [4.245996,  5.707715 ],
  [1.6115398, 2.0591364],
  [2.673064,  3.5667632]
]


col = 5
row = 7
anchor_idx = 3
b1 = [-2.7080502,   1.36687628, -0.11428463, -0.21773066]
b2 = [-2.573484 ,   1.4726449,  -0.06526405, -0.18472552]
#[[0.4326923076923077, 0.31971153846153844, 0.11298076923076923, 0.14182692307692307]]
#[(0.3506549942026817, 0.3963280057591697, 0.11335400314958731, 0.1418842224024299)]

num_cells = 13
is_apply_sigmoid = True

print(reconstruct_box(b1, col, row, anchors[anchor_idx], num_cells, is_apply_sigmoid=is_apply_sigmoid))
print(reconstruct_box(b2, col, row, anchors[anchor_idx], num_cells, is_apply_sigmoid=is_apply_sigmoid))