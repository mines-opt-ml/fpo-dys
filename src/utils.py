'''
Utilities for testing Differentiable shortest path algorithms
Daniel McKenzie
November 2022
'''
import torch
import torch.nn as nn 
import itertools
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split, Subset
from src.torch_Dijkstra import Dijkstra

## Custom initialization
def uniform_init(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.uniform_(m.weight, 1e-4, 1e-2)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.fill_(0.01)

## Small utility for rounding coordinates of points
def round_coordinates(vertex_name):
  vertex_coord = [int(vertex_name[0]), int(vertex_name[1])]
  return vertex_coord

## Utility for converting path in edge list format to node list format
def edge_to_node(path, edge_list, m, device):
  # Note: takes one sample path at a time!? 
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

def node_to_edge(paths, edge_list, four_neighbors=False):
  # converts paths to edges
  # assumes paths is shape (batch_size, m, m)
  # assumes edge_list is list of edges

  num_edges = len(edge_list)
  batch_size = paths.shape[0]
  grid_size = paths.shape[1]
  dijkstra = Dijkstra(grid_size=grid_size, euclidean_weight=True, four_neighbors=four_neighbors)
  
  edge_paths = torch.zeros(batch_size, num_edges)
  temp_costs = 1000. - 999*paths.numpy() #DM: Choice of constants here is a bit arbitrary.

  for i in range(batch_size):
    _, path_e = dijkstra.run_single(temp_costs[i,:,:],Gen_Data=True)
    path_e.reverse() # reverse list as output starts from bottom right corner
    
    # encode edge description of shortest path as a vector
    path_vec_e = torch.zeros(num_edges)
    for j in range(len(path_e)-1):
      path_vec_e[edge_list.index((path_e[j], path_e[j+1]))] = 1.

    assert path_vec_e.shape == edge_paths[i,:].shape
    edge_paths[i,:] = path_vec_e

  return edge_paths


def get_neighboring_vertices(curr_vertex, prev_vertex, m):
  '''
  Based on the around.py function in torch_Dijkstra, which is itself lightly adapted from code available at
https://github.com/google-research/google-research/blob/master/perturbations/experiments/shortest_path.py
  '''
  neighbors = []
  valid_offsets = itertools.product(range(-1, 2), range(-1, 2))
  for offset in valid_offsets:
    if abs(offset[0]) != abs(offset[1]):
      if (-1 < curr_vertex[0] + offset[0]< m) and (-1 < curr_vertex[1] + offset[1]< m):
        if (curr_vertex[0] + offset[0],curr_vertex[1] + offset[1]) != prev_vertex:
          if (curr_vertex[0] + offset[0],curr_vertex[1] + offset[1]) != curr_vertex:
            neighbors.append((curr_vertex[0] + offset[0],curr_vertex[1] + offset[1]))
  return neighbors

def compute_accuracy(pred_batch, true_batch, true_cost, edge_list, grid_size, device='cpu', pred_batch_edge_form=True):
  '''
  Simple utility for determining what fraction of predicted paths in pred_batch have the same (optimal) costs as the
  truth paths in true_batch. More sophisticated approaches could use Dijkstra's algorithm, but we find this suffices.
  Assumes true_cost and true_batch are in vertex form. But pred_batch might be in edge_form depending on the solver.
  '''
   
  score = 0.
  cost_pred_batch = 0.
  cost_true_batch = 0.
  batch_size = pred_batch.shape[0]
  pred_batch_prev = torch.zeros(12,12).to(device)
  for i in range(batch_size):
    pred_batch_i = edge_to_node(pred_batch[i,:], edge_list, grid_size, device)
    print('\n Predicted path:\n ')
    print(pred_batch_i)
    pred_batch_prev = pred_batch_i
    print('\n')
    cost_pred = torch.sum(true_cost[i,:,:] * pred_batch_i)
    cost_true = torch.sum(true_cost[i,:,:] * true_batch[i,:,:]) # assumes true batch is in vertex mode 
    # print('Cost matrix is ')
    # print(true_cost[i,:,:] * pred_batch_i)
    # print('\n true path is:\n ')
    # print(true_batch[i,:,:])
    # print('\n')
    print('\n cost predicted is '+ str(cost_pred.item())+ ' and true cost is '+ str(cost_true.item()))
    assert(true_cost[i,:,:].shape==(grid_size,grid_size))
    assert(pred_batch_i.shape==true_cost[i,:,:].shape)
    assert( true_batch[i,:,:].shape==true_cost[i,:,:].shape)

    cost_pred_batch += cost_pred
    cost_true_batch += cost_true
    if torch.abs(cost_pred - cost_true) < 1e-2:
      score += 1.

  return score/batch_size, cost_pred_batch/batch_size, cost_true_batch/batch_size



def compute_perfect_path_acc(pred_batch, true_batch):
  '''
  Simple utility for determining what fraction of predicted paths in pred_batch match the ground
  truth paths in true_batch. More sophisticated approaches could use Dijkstra's algorithm, but we find this suffices.
  '''
  score = 0.
  batch_size = pred_batch.shape[0]
  for i in range(batch_size):
    if torch.linalg.norm(torch.round(pred_batch[i,:]) - true_batch[i,:]) < 1e-2:
      score += 1.
  
  return score/batch_size

def compute_perfect_path_acc_vertex(pred_batch, true_batch):
  score = 0.
  batch_size = pred_batch.shape[0]
  for i in range(batch_size):
    path_map = pred_batch[i,:]
    true_map = true_batch[i,:]
    if torch.linalg.norm(path_map - true_map) < 1e-2:
      score += 1.
  
  return score/batch_size

## Utility for computing normalized regret 
# TODO: Fix this so that it can also be used with warcraft example.
def compute_regret_shortest_path(WW,d_batch, true_batch, pred_batch, type, edge_list, grid_size, device):
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
    dijkstra = Dijkstra(grid_size=m, euclidean_weight=True,four_neighbors=True)
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
        