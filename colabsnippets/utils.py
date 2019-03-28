import json
import random

def load_json(json_file_path):
  with open(json_file_path) as json_file:
    return json.load(json_file)

def shuffle_array(arr):
  arr_clone = arr[:]
  random.shuffle(arr_clone)
  return arr_clone