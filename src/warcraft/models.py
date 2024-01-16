import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer 
from src.dys_opt_net import DYS_opt_net
from pyepo.model.grb import shortestPathModel
from torchvision.models import resnet18
from src.shortest_path.shortest_path_utils import shortestPathModel_8

class WarcraftShortestPathNet(DYS_opt_net):
    def __init__(self, grid_size, A, b, edges, context_size, device='mps'):
        super(WarcraftShortestPathNet, self).__init__(A, b, device)
        self.grid_size = grid_size
        ## These layers are like resnet18
        resnet = resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.block = resnet.layer1
        # now convert to 1 channel
        self.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        # max pooling
        self.maxpool2 = nn.AdaptiveMaxPool2d((grid_size, grid_size))

        ## Optimization layer. Can be used within test_time_forward
        self.shortest_path_solver = shortestPathModel_8((self.grid_size, self.grid_size))

    def data_space_forward(self, d):
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
        return cost_vec
    
    def F(self, z, cost_vec):
        return cost_vec + 0.0005*z
    
    def test_time_forward(self, d):
        return self.data_space_forward(d)
    
