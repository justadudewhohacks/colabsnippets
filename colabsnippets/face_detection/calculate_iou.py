def calculate_iou(box0, box1):
  x0, y0, w0, h0 = box0
  x1, y1, w1, h1 = box1

  inter_ul_x, inter_ul_y = max(x0, x1), max(y0, y1)
  inter_br_x, inter_br_y = (min(x0 + w0, x1 + w1), min(y0 + h0, y1 + h1))
  inter_w, inter_h = inter_br_x - inter_ul_x, inter_br_y - inter_ul_y

  area0 = w0 * h0
  area1 = w1 * h1
  inter_area = inter_w * inter_h

  return inter_area / float(area0 + area1 - inter_area)