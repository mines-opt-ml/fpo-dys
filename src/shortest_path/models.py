import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer 
from src.dys_opt_net import DYS_opt_net
from pyepo.model.grb import shortestPathModel

class ShortestPathNet(DYS_opt_net):
  def __init__(self, A, b, edges, context_size, device='mps'):
    super(ShortestPathNet, self).__init__(A, b)
    self.context_size = context_size
    self.num_vertices = A.shape[0]
    self.num_edges = len(edges)
    self.hidden_dim = 2*context_size
    self.edges = edges
    self.device=device
    
    ## initialize fc layers
    self.fc_1 = nn.Linear(context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.num_edges)
    self.leaky_relu = nn.LeakyReLU(0.1)

    # initialize combinatorial solver
    self.shortest_path_solver = shortestPathModel((self.num_vertices, self.num_vertices))


  def F(self, z, cost_vec):
    '''
    gradient of cost vector with a little bit of regularization.
    '''
    return cost_vec + 0.0005*z

  def data_space_forward(self, d):
    z = self.leaky_relu(self.fc_1(d))
    cost_vec = self.fc_2(z)
    return cost_vec # size = num_edges
  
  def test_time_forward(self, d):
    '''
    Trying something different with test_time_forward: just returning cost vec
    so that we can use pyepo to evaluate.
    '''
    return self.data_space_forward(d)
  
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
  
# For use with blackbox-backprop, perturbed differentiable optimization etc.
# will use the pyEPO implementations of these schemes
  
class Generic_ShortestPathNet(nn.Module):
  def __init__(self, A, context_size, device = 'mps'):
    self.num_vertices = A.shape[0]
    self.num_edges = A.shape[1]
    self.hidden_dim = 2*context_size
    self.device = device

    ## Standard layers
    self.fc_1 = nn.Linear(context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.num_edges)
    self.leaky_relu = nn.LeakyReLU(0.1)

    # initialize combinatorial solver
    self.shortest_path_solver = shortestPathModel((self.num_vertices, self.num_vertices))
    

def forward(self, d):
  w = self.leaky_relu(self.fc_1(d))
  w = self.fc_2(w)
  # If in training mode, return only the predicted values.
  # If in testing mode, solve problem using gurobi.
  if self.training:
    return w
  else:
    batch_size = w.shape[0]
    solutions = torch.zeros(w.shape, device=self.device)
    for i in range(batch_size):
        self.shortest_path_solver.setObj(w[i,:])
        solution, _ = self.shortest_path_solver.solve()
        solutions[i,:] = torch.tensor(solution).to(self.device)
    return solutions



