"""
Implementation of Dijkstra for Pytorch. Adapted from Tensorflow version available at 
https://github.com/google-research/google-research/blob/master/perturbations/experiments/shortest_path.py

"""

import numpy as np
import torch
import heapq
import itertools


class Dijkstra:
  """Shortest path on a grid using Dijkstra's algorithm."""

  def __init__(
      self, grid_size, vertex_mode=True, edge_list=None, four_neighbors=False, initial_cost=1e10, euclidean_weight=False):
    self.four_neighbors = four_neighbors
    self.initial_cost = initial_cost
    self.euclidean_weight = euclidean_weight
    self.vertex_mode = vertex_mode
    self.edge_list = edge_list
    self.grid_size = grid_size

  def inside(self, x, y):
    return 0 <= x < self.shape[0] and 0 <= y < self.shape[1]

  def valid_move(self, x, y, off_x, off_y):
    size = np.sum(np.abs([off_x, off_y]))
    return ((size > 0) and
            (not self.four_neighbors or np.abs(size - 1)< 0.01) and
            self.inside(x + off_x, y + off_y))

  def around(self, x, y):
    coords = itertools.product(range(-1, 2), range(-1, 2))
    result = []
    for offset in coords:
      if self.valid_move(x, y, offset[0], offset[1]):
        result.append((x + offset[0], y + offset[1]))
    return result

  def reset(self):
    """Resets the variables to compute the shortest path of a cost matrix."""
    self.shape = [self.grid_size, self.grid_size]# costs.shape
    self.start = (0, 0)
    self.path_list = []  # Custom. To ensure compatibility 
    # of training data with DYS approach.
    self.end = (self.shape[0] - 1, self.shape[1] - 1)

    self.solution = np.ones(self.shape) * self.initial_cost
    self.solution[self.start] = 0.0

    self.queue = [(0.0, self.start)]
    self.visits = set(self.start)
    self.moves = dict()

    self.path = np.zeros(self.shape)
    self.path[self.start] = 1.0
    self.path[self.end] = 1.0

  def run_single(self, costs, Gen_Data = False):
    """Computes the shortest path on a single cost matrix."""
    self.reset()
    while self.queue:
      _, (x, y) = heapq.heappop(self.queue)
      if (x, y) in self.visits:
        continue
      for nx, ny in self.around(x, y):
        if (nx, ny) in self.visits:
          continue

        if self.euclidean_weight:
          weight = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
        else:
          weight = 1.0
        if self.vertex_mode:
          new_cost = weight * costs[nx, ny] + self.solution[x, y]
        else:
          edge_index = self.edge_list.index(((x + 0.5, y + 0.5), (nx + 0.5, ny + 0.5)))
          new_cost = weight * costs[edge_index] + self.solution[x, y]
        if new_cost < self.solution[nx, ny]:
          self.solution[nx, ny] = new_cost
          self.moves[(nx, ny)] = (x, y)
          heapq.heappush(self.queue, (new_cost, (nx, ny)))
      self.visits.add((x, y))

    curr = self.end
    # print(self.end)
    # print(curr)
    self.path_list.append((curr[0]+0.5, curr[1]+ 0.5))
    # print(self.path_list)
    while curr != self.start:
      curr = self.moves[curr]
      self.path[curr] = 1.0
      self.path_list.append((curr[0]+0.5, curr[1]+ 0.5))
      # print(self.path_list)
    
    if Gen_Data:
      return self.path, self.path_list
    else:
      return torch.from_numpy(self.path).float()

  def run_batch(self, tensor, Gen_Data):
    return torch.stack([self.run_single(tensor[i,:], Gen_Data)
                     for i in range(tensor.shape[0])],
                    axis=0)

  def __call__(self, tensor, batch_mode = False, Gen_Data=False):
    if len(tensor.shape) > 3:
      return torch.stack([self.run_batch(tensor[i])
                       for i in range(tensor.shape[0])],
                      axis=0)
    if batch_mode: # len(tensor.shape) == 3:
      return self.run_batch(tensor, Gen_Data)
    return self.run_single(tensor, Gen_Data)
