'''
Utilities for testing Differentiable shortest path algorithms
Daniel McKenzie
November 2022
'''
import numpy as np
import torch
import itertools
import torch.nn as nn

## Small utility for rounding coordinates of points
def RoundCoords(vertex_name):
  vertex_coord = [int(vertex_name[0]), int(vertex_name[1])]
  return vertex_coord

## Utility for converting path in edge list format to node list format
def Edge_to_Node(path, Edge_list, m, device):
  node_map = torch.zeros((m,m), device=device)
  node_map[0,0] = 1.
  node_map[-1,-1] = 1.
  path_list = []
  for edge_num, edge_val in enumerate(path):
    if edge_val > 0:
      edge_name = Edge_list[edge_num]
      path_list.append(edge_name)
      node_0 = RoundCoords(edge_name[0])
      node_1 = RoundCoords(edge_name[1])
      node_map[node_0[0], node_0[1]] += edge_val
      node_map[node_1[0], node_1[1]] += edge_val
  return node_map/2

## Utility for converting path in vertex format to edge list format
def Node_to_Edge(path, Edge_list):
    num_edges = len(Edge_list)
    path_e = np.zeros(num_edges)
    row_inds, col_inds = np.nonzero(path)
    for i in range(len(row_inds)-1):
        edge = ((row_inds[i]+0.5, col_inds[i]+0.5), (row_inds[i+1]+0.5, col_inds[i+1]+0.5))
        try:
          path_e[Edge_list.index(edge)] = 1.
        except:
          print(path)
    return path_e

def Next_Vertices(curr_vertex, prev_vertex, m):
  neighbors = []
  valid_offsets = itertools.product(range(-1, 2), range(-1, 2))
  for offset in valid_offsets:
    if abs(offset[0]) != abs(offset[1]):
      if (-1 < curr_vertex[0] + offset[0]< m) and (-1 < curr_vertex[1] + offset[1]< m):
        if (curr_vertex[0] + offset[0],curr_vertex[1] + offset[1]) != prev_vertex:
          if (curr_vertex[0] + offset[0],curr_vertex[1] + offset[1]) != curr_vertex:
            neighbors.append((curr_vertex[0] + offset[0],curr_vertex[1] + offset[1]))
  return neighbors

## Utility for computing the fraction of inferences for which the predicted
# path is optimal.

def Greedy_Decoder(node_map, m):
  curr_vertex = (0,0)
  prev_vertex = (-1, -1)
  path_map = torch.zeros(node_map.shape)
  path_map[0, 0] = 1.0
  visited_list = [(0,0)]
  count = 0
  while curr_vertex != (m-1, m-1) and count <= 1000:
    neighbors = Next_Vertices(curr_vertex, prev_vertex, m)
    next_vertex = curr_vertex
    next_Vertex_val = 0.0
    for neighbor in neighbors:
      if node_map[neighbor[0], neighbor[1]] > next_Vertex_val and neighbor not in visited_list: 
        next_Vertex_val = node_map[neighbor[0], neighbor[1]]
        next_vertex = neighbor
    prev_vertex = curr_vertex
    curr_vertex = next_vertex
    visited_list.append(curr_vertex)
    path_map[curr_vertex[0], curr_vertex[1]] = 1.0
    count += 1

  return path_map
    

def compute_perfect_path_acc(pred_batch, true_batch, Edge_list, grid_size, device):
  '''
  Greedily decode Node map from Edge_to_Node. Compute accuracy.
  '''
  score = 0.
  batch_size = pred_batch.shape[0]
  for i in range(batch_size):
    curr_map = Edge_to_Node(pred_batch[i,:], Edge_list, grid_size, device)
    true_map = Edge_to_Node(true_batch[i,:], Edge_list, grid_size, device)
    path_map = Greedy_Decoder(curr_map, grid_size).to(device)
    if torch.linalg.norm(path_map - true_map) < 0.001:
      score += 1.
  
  return score/batch_size

def compute_perfect_path_acc_vertex(pred_batch, true_batch):
  score = 0.
  batch_size = pred_batch.shape[0]
  for i in range(batch_size):
    path_map = pred_batch[i,:]
    true_map = true_batch[i,:]
    if torch.linalg.norm(path_map - true_map) < 0.001:
      score += 1.
  
  return score/batch_size

## Utility for computing normalized regret 
def compute_regret(WW,d_batch, true_batch, pred_batch, type, Edge_list, grid_size, device):
  '''
  Computes the difference in length between predicted path and best path.
  '''
  WW = WW.to(device)
  true_weights = torch.transpose(torch.matmul(WW, torch.transpose(d_batch, 0, 1)), 0, 1)
  regret = 0.
  batch_size = pred_batch.shape[0]

  # print('pred_batch.shape = ', pred_batch.shape)
  for i in range(batch_size):
    if type == "E":
      curr_map = Edge_to_Node(pred_batch[i,:], Edge_list, grid_size, device)
      true_map = Edge_to_Node(true_batch[i,:], Edge_list, grid_size, device)
      path_map = Greedy_Decoder(curr_map, grid_size).to(device)
    else:
      path_map = pred_batch[i,:]  
      true_map = Edge_to_Node(true_batch[i,:], Edge_list, grid_size, device)
    
    length_shortest_path = torch.dot(true_map.view(grid_size**2), true_weights[i,:])
    difference = path_map - true_map
    temp_regret = torch.dot(difference.view(grid_size**2), true_weights[i,:])
    # In rare cases, the network fails to predict a path traversing from top-left to
    # bottom-right corner. In this case, the length of the predicted path can be shorter
    # than the true_path. These cases are extremely rare, we assign them a regret equal
    # to the length of the true path
    if temp_regret < 0:
      regret += 1
    else:
      regret += temp_regret/length_shortest_path
  return regret/batch_size

#### The following are utilities related to Regret loss.
def RegretLoss(nn.Module):
   def __init__(self, n, device):
        super(RegretLoss, self).__init__()
        self.device = device
        self.n = n  # number of variables
    
    def forward(self, d, w_true, x_pred):
      '''
      d is (batch of) contexts, w_true is (batch of) true 
      cost vectors.
      '''
      cost = torch.matmul(w_true.view(-1, self.n), x_pred)
      return torch.mean(cost)