import torch
import torch.nn as nn
import cvxpy as cp
# import blackbox_backprop as bb 
from cvxpylayers.torch import CvxpyLayer
from abc import ABC, abstractmethod
from src.shortest_path.dYS_opt_net import DYS_opt_net
from src.shortest_path.utils import Edge_to_Node
from src.shortest_path.torch_Dijkstra import Dijkstra
from src.shortest_path import perturbations



## Create NN using DYS layer. Look how easy it is!
class ShortestPathNet(DYS_opt_net):
  def __init__(self, A, b, num_vertices, num_edges, Edges, context_size, device='cpu'):
    super(ShortestPathNet, self).__init__(A, b)
    self.context_size = context_size
    self.num_vertices = num_vertices
    self.num_edges = num_edges
    self.hidden_dim = 2*context_size
    self.Edges = Edges
    self.device=device

    ## Compute geometric edge length multiplier
    # Edge_lengths = []
    #for edge in Edges:
    #  v1 = edge[0]
    #  v2 = edge[1]
    #  edge_length = torch.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    #  Edge_lengths.append(edge_length)

    # self.Edge_lengths = torch.stack(Edge_lengths)
    ## initialize fc layers
    self.fc_1 = nn.Linear(context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.num_edges)
    self.leaky_relu = nn.LeakyReLU(0.1)


  def F(self, z, cost_vec):
    '''
    gradient of cost vector with a little bit of regularization.
    '''
    return cost_vec + 0.0005*z

  def data_space_forward(self, d):
    z = self.leaky_relu(self.fc_1(d))
    cost_vec = self.fc_2(z)
    return cost_vec

## Create NN using cvxpylayers
class Cvx_ShortestPathNet(nn.Module):
  def __init__(self, A, b, context_size, device='cpu'):
    super().__init__()
    self.b = b.to(device)
    self.A = A.to(device)
    self.n1 = A.shape[0]
    self.n2 = A.shape[1]
    self.device = device
    self.hidden_dim = 2*context_size

    ## Standard layers
    self.fc_1 = nn.Linear(context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.n2)
    self.leaky_relu = nn.LeakyReLU(0.1)
    
    ## cvxpy layer
    x = cp.Variable(self.n2)
    w = cp.Parameter(self.n2)
    AA = cp.Parameter((self.n1, self.n2))
    bb = cp.Parameter(self.n1)

    objective = cp.Minimize(w.T@x + 0.5*cp.sum_squares(x))
    constraints = [AA@x == bb, x >=0]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    self.cvxpylayer = CvxpyLayer(problem, parameters=[AA, bb, w], variables=[x])
    
  ## Put it all together
  def forward(self, d):
    w = self.leaky_relu(self.fc_1(d))
    w = self.fc_2(w)
    solution, = self.cvxpylayer(self.A, self.b, w)
    return solution

## Create NN using perturbed differentiable optimization
class Pert_ShortestPathNet(nn.Module):
    '''
    This net is equipped to run an m-by-m grid graphs. No A matrix is necessary.
    '''
    def __init__(self, m, context_size, device='cpu'):
        super().__init__()
        self.m = m
        self.device = device
        self.hidden_dim = 2*context_size

        ## Standard layers
        self.fc_1 = nn.Linear(context_size, self.hidden_dim)
        self.fc_2 = nn.Linear(self.hidden_dim, self.m**2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        ## Perturbed Differentiable Optimization layer
        dijkstra = Dijkstra(euclidean_weight=True,four_neighbors=True)
        self.dijkstra = dijkstra
        self.pert_dijkstra = perturbations.perturbed(dijkstra,
                                      num_samples=3,
                                      sigma=1.0,
                                      noise='gumbel',
                                      batched=True,
                                      device=self.device)
        
      ## Put it all together
    def forward(self, d):
        w = self.leaky_relu(self.fc_1(d))
        w = self.fc_2(w)
        if self.training:
          path = self.pert_dijkstra(w.view(w.shape[0], self.m, self.m))
        else:
          path = self.dijkstra(w.view(w.shape[0], self.m, self.m))
        return path.to(self.device)
          
## Create NN using Blackbox backprop of Vlastelica et al
class BB_ShortestPathNet(nn.Module):
    '''
    This net is equipped to run an m-by-m grid graphs. No A matrix is necessary.
    Not quite working. No signal is backpropagating?
    '''
    def __init__(self, m, context_size, device='cpu'):
        super().__init__()
        self.m = m
        self.device = device
        self.hidden_dim = 2*context_size
        self.shortestPath = bb.ShortestPath()

        ## Standard layers
        self.fc_1 = nn.Linear(context_size, self.hidden_dim)
        self.fc_2 = nn.Linear(self.hidden_dim, self.m**2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, d):
        w = self.leaky_relu(self.fc_1(d))
        w = self.fc_2(w)
        suggested_weights = w.view(w.shape[0], self.m, self.m)
        suggested_shortest_paths = self.shortestPath.apply(suggested_weights, 100)
        
        return suggested_shortest_paths
        
        
