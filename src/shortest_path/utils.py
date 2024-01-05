import torch

## Utility for converting path in edge list format to node list format
def edge_to_node(path, edges, grid_size, device):
  # Note: takes one sample path at a time!? 
  node_map = torch.zeros((grid_size,grid_size), device=device)
  node_map[0,0] = 1.
  node_map[-1,-1] = 1.
  path_list = []
  for edge_num, edge_val in enumerate(path):
    if edge_val > 0:
      edge_name = edges[edge_num]
      path_list.append(edge_name)
      node_0 = edge_name[0]
      node_1 = edge_name[1]
      node_map[int(node_0/grid_size), node_0 % grid_size] += edge_val
      node_map[int(node_1/grid_size), node_1 % grid_size] += edge_val
  return node_map/2