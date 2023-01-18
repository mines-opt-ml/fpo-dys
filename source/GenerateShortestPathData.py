import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split, Subset
from source.torch_Dijkstra import Dijkstra
from source.utils import Node_to_Edge

def create_shortest_path_data(m, train_size, test_size, context_size):
    '''
     Function generates shortest path training problem for m-by-m grid graph.
    '''
    # Construct vertices. Note each vertex is at the middle of the
    # square.
    Vertices = []
    for i in range(m):
        for j in range(m):
            Vertices.append((i+0.5, j+0.5))

    num_vertices = len(Vertices)
    # Construct edges.
    Edges = []
    for i, v1 in enumerate(Vertices):
        for j, v2 in enumerate(Vertices):
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
    A = torch.zeros((len(Vertices), len(Edges)))
    for j, e in enumerate(Edges):
        ind0 = Vertices.index(e[0])
        ind1 = Vertices.index(e[1])
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
# train_dataset_v, test_dataset_v = random_split(dataset_v, [train_size, test_size], generator=torch.Generator().manual_seed(42))
#  train_dataset_e, test_dataset_e = random_split(dataset_e, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    return train_dataset_v, test_dataset_v, train_dataset_e, test_dataset_e, WW, A.float(), b.float(), num_edges, Edges
        
