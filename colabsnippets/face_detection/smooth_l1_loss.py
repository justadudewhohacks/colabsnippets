import keras.backend as K


def smooth_l1_loss(y_true, y_pred, HUBER_DELTA=0.5):
  x = K.abs(y_true - y_pred)
  x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
  return x
