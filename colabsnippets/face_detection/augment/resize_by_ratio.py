import cv2


def resize_by_ratio(img, boxes, landmarks, sx, sy):
  img = cv2.resize(img, (int(sx * img.shape[1]), int(sy * img.shape[0])))
  boxes = [[sx * bx, sy * by, bw * sx, bh * sy] for bx, by, bw, bh in boxes]
  landmarks = [[[sx * lx, sy * ly] for lx, ly in l] for l in landmarks]
  return img, boxes, landmarks
