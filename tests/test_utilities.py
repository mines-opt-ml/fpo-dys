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
    
    def test_edge_to_node(self):
        # Test that edge_to_node returns correct node path

        ## Set device
        device = 'cpu'
        print('device: ', device)


        grid_size = 12
        base_data_path = os.path.join(os.path.dirname(__file__), '../src/warcraft/warcraft_data/')

        ## Load data
        data_path = base_data_path + 'Warcraft_training_data'+str(grid_size)+'.pth'
        state = torch.load(data_path)

        ## Extract data from state
        train_dataset_e = state['train_dataset_e']
        test_dataset_e = state['test_dataset_e']
        train_dataset_v = state['train_dataset_v']
        test_dataset_v = state['test_dataset_v']

        grid_size = state["m"]
        A = state["A"].float()
        b = state["b"].float()
        num_edges = state["num_edges"]
        edge_list = state["edge_list"]
        edge_list_torch = torch.tensor(edge_list)

        A = A.to(device)
        b = b.to(device)
        n_samples_train = len(train_dataset_e)
        d_batch_e, path_batch_e = train_dataset_e[0:n_samples_train]
        d_batch_v, path_batch_v = train_dataset_v[0:n_samples_train]

        for i in range(n_samples_train):
            path_batch_v2_i = edge_to_node(path_batch_e[i,:], edge_list, grid_size, device='cpu')
            self.assertTrue( torch.allclose(path_batch_v2_i, path_batch_v[i,:,:]) )

if __name__ == '__main__':
    unittest.main()