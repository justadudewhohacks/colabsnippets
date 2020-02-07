def filter_abs_boxes_out_of_borders(abs_boxes, hw):
  filtered_boxes = []
  for box in abs_boxes:
    x0, y0, w, h = box
    x1, y1 = x0 + w, y0 + h

    num_out_of_img = 0
    corners = [(x0, y0), (x1, y1), (x0, y1), (x1, y0)]
    for x, y in corners:
      if x < 0 or y < 0 or x > hw[1] or y > hw[0]:
        num_out_of_img += 1

    if num_out_of_img < len(corners):
      filtered_boxes.append(box)

  return filtered_boxes


def filter_abs_landmarks_out_of_borders(abs_landmarks, hw):
  filtered_landmarks = []
  for l in abs_landmarks:
    num_out_of_img = 0
    for x, y in l:
      if x < 0 or y < 0 or x > hw[1] or y > hw[0]:
        num_out_of_img += 1
    if num_out_of_img < len(l):
      filtered_landmarks.append(l)
  return filtered_landmarks