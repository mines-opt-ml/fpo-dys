'''
Utilities for testing Differentiable shortest path algorithms
Daniel McKenzie
November 2022
'''
import numpy as np
import torch
import itertools
import torch.nn as nn

import networkx as nx
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split, Subset
from src.torch_Dijkstra import Dijkstra

## Small utility for rounding coordinates of points
def round_coordinates(vertex_name):
  vertex_coord = [int(vertex_name[0]), int(vertex_name[1])]
  return vertex_coord

## Utility for converting path in edge list format to node list format
def edge_to_node(path, edge_list, m, device):
  node_map = torch.zeros((m,m), device=device)
  node_map[0,0] = 1.
  node_map[-1,-1] = 1.
  path_list = []
  for edge_num, edge_val in enumerate(path):
    if edge_val > 0:
      edge_name = edge_list[edge_num]
      path_list.append(edge_name)
      node_0 = round_coordinates(edge_name[0])
      node_1 = round_coordinates(edge_name[1])
      node_map[node_0[0], node_0[1]] += edge_val
      node_map[node_1[0], node_1[1]] += edge_val
  return node_map/2

# ## Utility for converting path in vertex format to edge list format (BUGGY DOES NOT WORK)
# def node_to_edge(path, edge_list):
#     num_edges = len(edge_list)
#     path_e = np.zeros(num_edges)
#     row_inds, col_inds = np.nonzero(path)
#     for i in range(len(row_inds)-1):
#         edge = ((row_inds[i]+0.5, col_inds[i]+0.5), (row_inds[i+1]+0.5, col_inds[i+1]+0.5))
#         try:
#           path_e[edge_list.index(edge)] = 1.
#         except:
#           print('error!')
#           print(path)
#     return path_e

def node_to_edge(paths, edge_list):
  # converts paths to edges
  # assumes paths is shape (batch_size, m, m)
  # assumes edge_list is list of edges
  dijkstra = Dijkstra(euclidean_weight=True,four_neighbors=False)

  edge_paths = torch.zeros(paths.shape[0], len(edge_list))

  num_edges = len(edge_list)
  batch_size = paths.shape[0]
  temp_costs = 2 - paths.numpy()

  for i in range(batch_size):
    path_v, path_e = dijkstra.run_single(temp_costs[i,:,:],Gen_Data=True)
    path_e.reverse() # reverse list as output starts from bottom right corner
    
  # encode edge description of shortest path as a vector
  path_vec_e = torch.zeros(len(edge_list))
  for i in range(len(path_e)-1):
    path_vec_e[edge_list.index((path_e[i], path_e[i+1]))] = 1

    assert path_vec_e.shape == edge_paths[i,:].shape
    edge_paths[i,:] = path_vec_e

  return edge_paths


def get_neighboring_vertices(curr_vertex, prev_vertex, m):
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

def greedy_decoder(node_map, m):
  curr_vertex = (0,0)
  prev_vertex = (-1, -1)
  path_map = torch.zeros(node_map.shape)
  path_map[0, 0] = 1.0
  visited_list = [(0,0)]
  count = 0
  while curr_vertex != (m-1, m-1) and count <= 1000:
    neighbors = get_neighboring_vertices(curr_vertex, prev_vertex, m)
    next_vertex = curr_vertex
    next_vertex_val = 0.0
    for neighbor in neighbors:
      if node_map[neighbor[0], neighbor[1]] > next_vertex_val and neighbor not in visited_list: 
        next_vertex_val = node_map[neighbor[0], neighbor[1]]
        next_vertex = neighbor
    prev_vertex = curr_vertex
    curr_vertex = next_vertex
    visited_list.append(curr_vertex)
    path_map[curr_vertex[0], curr_vertex[1]] = 1.0
    count += 1

  return path_map
    

def compute_perfect_path_acc(pred_batch, true_batch, edge_list, grid_size, device):
  '''
  Greedily decode Node map from Edge_to_Node. Compute accuracy.
  '''
  score = 0.
  batch_size = pred_batch.shape[0]
  for i in range(batch_size):
    curr_map = edge_to_node(pred_batch[i,:], edge_list, grid_size, device)
    true_map = edge_to_node(true_batch[i,:], edge_list, grid_size, device)
    path_map = greedy_decoder(curr_map, grid_size).to(device)
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
def compute_regret(WW,d_batch, true_batch, pred_batch, type, edge_list, grid_size, device):
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
      curr_map = edge_to_node(pred_batch[i,:], edge_list, grid_size, device)
      true_map = edge_to_node(true_batch[i,:], edge_list, grid_size, device)
      path_map = greedy_decoder(curr_map, grid_size).to(device)
    else:
      path_map = pred_batch[i,:]  
      true_map = edge_to_node(true_batch[i,:], edge_list, grid_size, device)
    
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

# #### The following are utilities related to Regret loss.
# def RegretLoss(nn.Module):
#     def __init__(self, n, device):
#         super(RegretLoss, self).__init__()
#         self.device = device
#         self.n = n  # number of variables
    
#     def forward(self, d, w_true, x_pred):
#       '''
#       d is (batch of) contexts, w_true is (batch of) true 
#       cost vectors.
#       '''
#       cost = torch.matmul(w_true.view(-1, self.n), x_pred)
#       return torch.mean(cost)

def create_shortest_path_data(m, train_size, test_size, context_size):
    '''
     Function generates shortest path training problem for m-by-m grid graph.
    '''
    # Construct vertices. Note each vertex is at the middle of the
    # square.
    vertices = []
    for i in range(m):
        for j in range(m):
            vertices.append((i+0.5, j+0.5))

    num_vertices = len(vertices)
    # Construct edges.
    Edges = []
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            norm_squared = (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2
            if 0 < norm_squared < 1.01 and i < j:
                Edges.append((v1, v2))

    num_edges = len(Edges)
    
    ## Fix context --> edge weight matrix.
    WW = 10*torch.rand(num_vertices, context_size)
    
    ## Generate all contexts
    Context = torch.rand(train_size+test_size, context_size)

    ## Prepare to generate and store all paths.
    # Note each path is an m-by-m matrix.
    Paths_List_V = []  # All true shortest paths, in vertex form
    Paths_List_E = []  # All true shortest paths, in edge form

    ## Construct an instance of the dijkstra class
    # Only considering four neighbors---no diag edges
    dijkstra = Dijkstra(euclidean_weight=True,four_neighbors=True)
    # Loop and determine the shortest paths
    for context in enumerate(Context):
        weight_vec = torch.matmul(WW, context[1]) # context[1] as the enumerate command yields tuples
        Costs = weight_vec.view((m,m))
        
        # Use Dijkstra's algorithm to find shortest path from top left to 
        # bottom right corner
        path_v, path_e = dijkstra.run_single(Costs,Gen_Data=True)
        path_e.reverse() # reverse list as output starts from bottom right corner
        # place into the Paths_List
        Paths_List_V.append(torch.from_numpy(path_v))
        
        # Encode edge description of shortest path as a vector
        path_vec_e = torch.zeros(len(Edges))
        for i in range(len(path_e)-1):
            try:
                path_vec_e[Edges.index((path_e[i], path_e[i+1]))] = 1
            except:
                path_vec_e[Edges.index((path_e[i+1], path_e[i]))] = 1
                print('path ' + str(context[0]) + ' involves backtracking')
        Paths_List_E.append(path_vec_e)
        
    ## convert Paths_List to a tensor
    Paths_V = torch.stack(Paths_List_V)
    Paths_E = torch.stack(Paths_List_E)
    
    ## Create b vector necessary for LP approach to shortest path
    b = torch.zeros(m**2)
    b[0] = -1.0
    b[-1] = 1.0

    ## Vertex Edge incidence matrix
    A = torch.zeros((len(vertices), len(Edges)))
    for j, e in enumerate(Edges):
        ind0 = vertices.index(e[0])
        ind1 = vertices.index(e[1])
        A[ind0,j] = -1.
        A[ind1,j] = +1
        
    # Construct and return dataset.
    dataset_v = TensorDataset(Context.float(), Paths_V.float())
    dataset_e = TensorDataset(Context.float(), Paths_E.float())
    train_dataset_v = Subset(dataset_v, range(train_size))
    test_dataset_v = Subset(dataset_v, range(train_size, train_size + test_size))
    train_dataset_e = Subset(dataset_e, range(train_size))
    test_dataset_e = Subset(dataset_e, range(train_size, train_size + test_size))
    # avoid random_split as we want the (context, path) pairs to be in the same order
    # for the vertex (_v) and edge (_e) datasets.
    return train_dataset_v, test_dataset_v, train_dataset_e, test_dataset_e, WW, A.float(), b.float(), num_edges, Edges
        