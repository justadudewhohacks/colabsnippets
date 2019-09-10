import math

def get_box_center(rel_box):
  x, y, w, h = rel_box
  ct_x = x + (w / 2)
  ct_y = y + (h / 2)
  return ct_x, ct_y


def get_gt_coords(gt_rel_box, num_cells):
  col, row = get_cell_position_of_box(gt_rel_box, num_cells)
  ct_x, ct_y = get_box_center(gt_rel_box)
  gt_x = (ct_x * num_cells) - col
  gt_y = (ct_y * num_cells) - row
  return [gt_x, gt_y]

def get_cell_position_of_box(rel_box, num_cells):
  # TODO: what if box centers are out of grid?
  in_grid_range = lambda val, num_cells: min(num_cells - 1, max(0, val))
  ct_x, ct_y = get_box_center(rel_box)
  col = in_grid_range(math.floor(ct_x * num_cells), num_cells)
  row = in_grid_range(math.floor(ct_y * num_cells), num_cells)
  return col, row
