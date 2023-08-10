# Assume path is root directory
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import time as time
import unittest

import sys
root_dir = "../"
# sys.path.append(root_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# print('root directory = ')


from src.models import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
from src.models import DYS_Warcraft_Net, Pert_Warcraft_Net
from src.trainer import trainer
from src.utils import edge_to_node, node_to_edge

class test_edge_to_node(unittest.TestCase):

    def setUp(self):
         ## Set device
        self.device = 'cpu'
        print('device: ', self.device)

        grid_size=12
        base_data_path = os.path.join(os.path.dirname(__file__), '../src/warcraft/warcraft_data/')

        ## Load data
        data_path = base_data_path + 'Warcraft_training_data'+str(grid_size)+'.pth'
        state = torch.load(data_path)

        ## Extract data from state
        train_dataset_e = state['train_dataset_e']
        test_dataset_e = state['test_dataset_e']
        train_dataset_v = state['train_dataset_v']
        test_dataset_v = state['test_dataset_v']

        self.grid_size = state["m"]
        A = state["A"].float()
        b = state["b"].float()
        self.num_edges = state["num_edges"]
        self.edge_list = state["edge_list"]
        edge_list_torch = torch.tensor(self.edge_list)

        self.A = A.to(self.device)
        self.b = b.to(self.device)
        self.n_samples = len(train_dataset_e)
        self.d_edge, self.path_edge = train_dataset_e[0:self.n_samples]
        self.d_vertex, self.path_vertex = train_dataset_v[0:self.n_samples]
    
    def test_edge_to_node(self):
        # Test that edge_to_node returns correct node path

        for i in range(self.n_samples):
            path_vertex2 = edge_to_node(self.path_edge[i,:], self.edge_list, self.grid_size, device=self.device)
            self.assertTrue( torch.allclose(path_vertex2, self.path_vertex[i,:,:]))

            print('i = ', i, ', path_batch_v = edge_to_node(path_batch_e)')

            path_edge2 = node_to_edge(path_vertex2, self.edge_list, four_neighbors=False)
            self.assertTrue( torch.allclose( path_edge2, self.path_edge[i,:]) )

            print('i = ', i, ', path_batch_e = node_to_edge(path_batch_v)')


if __name__ == '__main__':
    unittest.main()