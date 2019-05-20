import math

inverse_sigmoid = lambda x: math.log(x / (1 - x))
sigmoid = lambda x: 1 / (1 + math.exp(-x))