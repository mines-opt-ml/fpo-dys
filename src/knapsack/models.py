#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:17:50 2022

@author: danielmckenzie

Models for the shortest path prediction problem.
"""

import torch
import torch.nn as nn
from src.dys_opt_net import DYS_opt_net
from pyepo.model.grb import knapsackModel
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer 



## Create NN using DYS layer. Look how easy it is!
class KnapSackNet(DYS_opt_net):
  def __init__(self, weights, capacities, num_constraints, num_resources, context_size, device):
    A1 = torch.cat([weights, torch.eye(num_constraints, device=device), torch.zeros(weights.shape, device=device)], dim=1)
    A2 = torch.cat([torch.eye(num_resources, device=device), torch.zeros((num_resources, num_constraints),device=device), torch.eye(num_resources,device=device)], dim=1)
    A = torch.cat([A1, A2],dim=0)
    b = torch.cat([capacities, torch.ones(num_resources,device=device)]) # torch.ones(num_constraints,device=device)
    super(KnapSackNet, self).__init__(A, b, device=device)
    self.weights = weights
    self.context_size = context_size
    self.num_constraints = num_constraints
    self.num_resources = num_resources
    self.hidden_dim = 100*context_size
    self.device = device
    self.zero_padding_dim = self.num_constraints + self.num_resources

    ## initialize fc layers
    self.fc_1 = nn.Linear(context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.fc_3 = nn.Linear(self.hidden_dim, self.num_resources)
    #self.fc_4 = nn.Linear(self.hidden_dim, self.num_resources)
    self.leaky_relu = nn.LeakyReLU(0.1)
    self.dropout = nn.Dropout(0.3)

    # initialize combinatorial solver
    self.knapsack_solver = knapsackModel(weights.cpu().detach().numpy(), capacities.cpu().detach().numpy())


  def F(self, z, cost_vec):
    '''
    gradient of cost vector with a little bit of regularization.
    NB: this is a max, not a min problem, hence the negative signs.
    '''
    return  -cost_vec + 0.05*z  # tried up to 0.5

  def data_space_forward(self, d):
    z = self.leaky_relu(self.fc_1(d))
    z = self.leaky_relu(self.fc_2(z))
    cost_vec = self.dropout(self.fc_3(z))
    if self.training:
      ## NB: Now pad cost_vec with zeros, to account for dummy variables
      batch_size = d.shape[0]
      zero_padding = torch.zeros((batch_size, self.zero_padding_dim), device=self.device)
      return torch.cat([cost_vec, zero_padding], dim=1) # cost_vec
    else:
      return cost_vec
  
  def test_time_forward(self, d):
    # w = self.data_space_forward(d)
    # batch_size = w.shape[0]
    # solutions = torch.zeros(w.shape, device=self.device)
    # for i in range(batch_size):
    #   self.knapsack_solver.setObj(w[i,:self.num_resources])
    #   solution, _ = self.knapsack_solver.solve()
    #   zero_padding = torch.zeros((self.zero_padding_dim), device=self.device)
    #   solutions[i,:] = torch.cat([torch.tensor(solution).to(self.device), zero_padding])
    # return solutions
    return self.data_space_forward(d)
    

## CVX approach
class Cvx_KnapsackNet(nn.Module):
  def __init__(self, weights, capacities, num_constraints, num_resources, context_size, device='mps'):
    super().__init__()
    A1 = torch.cat([weights, torch.eye(num_constraints, device=device), torch.zeros(weights.shape, device=device)], dim=1)
    A2 = torch.cat([torch.eye(num_resources, device=device), torch.zeros((num_resources, num_constraints),device=device), torch.eye(num_resources,device=device)], dim=1)
    A = torch.cat([A1, A2],dim=0)
    b = torch.cat([capacities, torch.ones(num_resources,device=device)]) # torch.ones(num_constraints,device=device)
    self.weights = weights
    self.context_size = context_size
    self.num_constraints = num_constraints
    self.num_resources = num_resources
    self.A = A.to(device)
    self.b = b.to(device)
    self.n1 = A.shape[0]
    self.n2 = A.shape[1]
    self.hidden_dim = 100*context_size
    self.device = device
    self.zero_padding_dim = self.num_constraints + self.num_resources

    ## Standard layers
    self.fc_1 = nn.Linear(context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.fc_3 = nn.Linear(self.hidden_dim, self.num_resources)
    self.leaky_relu = nn.LeakyReLU(0.1)
    self.dropout = nn.Dropout(0.3)

    # initialize combinatorial solver
    self.knapsack_solver = knapsackModel(weights.cpu().detach().numpy(), capacities.cpu().detach().numpy())

     ## cvxpy layer
    x = cp.Variable(self.n2)
    w = cp.Parameter(self.n2)
    AA = cp.Parameter((self.n1, self.n2))
    bb = cp.Parameter(self.n1)

    objective = cp.Minimize(-w.T@x + 0.5*cp.sum_squares(x))
    constraints = [AA@x == bb, x >=0]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    self.cvxpylayer = CvxpyLayer(problem, parameters=[AA, bb, w], variables=[x])
    
  ## Put it all together
  def forward(self, d):
    z = self.leaky_relu(self.fc_1(d))
    z = self.leaky_relu(self.fc_2(z))
    cost_vec = self.dropout(self.fc_3(z))
    if self.training:
      batch_size = d.shape[0]
      zero_padding = torch.zeros((batch_size, self.zero_padding_dim), device=self.device)
      w = torch.cat([cost_vec, zero_padding], dim=1)
      solution, = self.cvxpylayer(self.A, self.b, w)
      return solution
    else:
      return cost_vec


## Generic model for predicting weights. To be used with PyEPO benchmarks

class ValPredictNet(nn.Module):
  def __init__(self, num_knapsack, num_item, num_feat, weights, capacities, device):
    super().__init__()
    self.context_size = num_feat
    self.num_constraints = num_knapsack
    self.num_resources = num_item
    self.hidden_dim = 100*num_feat
    self.weights = weights
    self.capacities = capacities
    self.device = device

    ## initialize fc layers
    self.fc_1 = nn.Linear(self.context_size, self.hidden_dim)
    self.fc_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.fc_3 = nn.Linear(self.hidden_dim, self.num_resources)
    self.leaky_relu = nn.LeakyReLU(0.1)
    self.dropout = nn.Dropout(0.3)

    # initialize combinatorial solver
    self.knapsack_solver = knapsackModel(weights.cpu().detach().numpy(), capacities.cpu().detach().numpy())
    
  def forward(self, d):
    z = self.leaky_relu(self.fc_1(d))
    z = self.leaky_relu(self.fc_2(z))
    w = self.dropout(self.fc_3(z))
    return w



