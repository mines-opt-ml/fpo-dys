import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer 
from src.dys_opt_net import DYS_opt_net
from pyepo.model.grb import shortestPathModel
from torchvision.models import resnet18
from src.shortest_path.shortest_path_utils import shortestPathModel_8

class WarcraftShortestPathNet(DYS_opt_net):
    def __init__(self, grid_size, A, b, device='mps'):
        super(WarcraftShortestPathNet, self).__init__(A, b, device)
        self.grid_size = grid_size
        ## These layers are like resnet18
        resnet = resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool1 = resnet.maxpool
        self.block = resnet.layer1
        # now convert to 1 channel
        self.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        # max pooling
        self.maxpool2 = nn.AdaptiveMaxPool2d((grid_size, grid_size))
        ## add a dropout layer
        # self.dropout = nn.Dropout(0.3)

        ## Optimization layer. Can be used within test_time_forward
        self.shortest_path_solver = shortestPathModel_8((self.grid_size, self.grid_size))
        ## Compute geometric edge length multiplier
        # self.edge_lengths = torch.zeros(len(self.shortest_path_solver.edges), device=self.device)
        # for e, edge in enumerate(self.shortest_path_solver.edges):
        #     node_0_coords = self.shortest_path_solver.nodes_map[edge[0]]
        #     node_1_coords = self.shortest_path_solver.nodes_map[edge[1]]
        #     nodes_dist = torch.sqrt(torch.tensor((node_0_coords[0] - node_1_coords[0])**2,device=self.device) + torch.tensor((node_0_coords[1] - node_1_coords[1])**2,device=self.device))
        #     self.edge_lengths[e] = nodes_dist
        
    def _data_space_forward(self, d):
        h = self.conv1(d)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool1(h)
        h = self.block(h)
        h = self.conv2(h)
        out = self.maxpool2(h)
        # reshape for optmodel
        out = torch.squeeze(out, 1)
        cost_vec = out.reshape(out.shape[0], -1)
        if self.training:
            batch_size = cost_vec.shape[0]
            train_cost_vec = torch.zeros((batch_size, len(self.shortest_path_solver.edges)),device=self.device)
            for e, edge in enumerate(self.shortest_path_solver.edges):
                train_cost_vec[:,e] = cost_vec[:,edge[1]]
            return train_cost_vec
        else:
            return cost_vec
    
    def F(self, z, cost_vec):
        return cost_vec + 0.0005*z
    
    def test_time_forward(self, d):
        return self._data_space_forward(d)
    

class Cvx_WarcraftShortestPathNet(nn.Module):
    def __init__(self, grid_size, A, b, device='mps'):
        super().__init__()
        # Linear program variables
        self.A = A
        self.b = b
        self.n1 = A.shape[0]
        self.n2 = A.shape[1]
        self.grid_size = grid_size
        # device
        self.device = device
        # These layers are like resnet18
        resnet = resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool1 = resnet.maxpool
        self.block = resnet.layer1
        # now convert to 1 channel
        self.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        # max pooling
        self.maxpool2 = nn.AdaptiveMaxPool2d((grid_size, grid_size))\
        # Optimization layer. Can be used within test_time_forward
        self.shortest_path_solver = shortestPathModel_8((self.grid_size, self.grid_size))
        # cvxpy layer
        x = cp.Variable(self.n2)
        w = cp.Parameter(self.n2)
        AA = cp.Parameter((self.n1, self.n2))
        bb = cp.Parameter(self.n1)

        objective = cp.Minimize(w.T@x + 0.5*cp.sum_squares(x))
        constraints = [AA@x == bb, x >=0]
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        self.cvxpylayer = CvxpyLayer(problem, parameters=[AA, bb, w], variables=[x])
        
    def _data_space_forward(self, d):
        h = self.conv1(d)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool1(h)
        h = self.block(h)
        h = self.conv2(h)
        out = self.maxpool2(h)
        # reshape for optmodel
        out = torch.squeeze(out, 1)
        cost_vec = out.reshape(out.shape[0], -1)
        if self.training:
            batch_size = cost_vec.shape[0]
            train_cost_vec = torch.zeros((batch_size, len(self.shortest_path_solver.edges)),device=self.device)
            for e, edge in enumerate(self.shortest_path_solver.edges):
                train_cost_vec[:,e] = cost_vec[:,edge[1]]
            return train_cost_vec
        else:
            return cost_vec
        
     # Put it all together
    def forward(self, d):
        w = self._data_space_forward(d)
        if self.training:
            solution, = self.cvxpylayer(self.A, self.b, w)
            return solution
        else:
            return w
