# Assume path is root directory
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import time as time
import unittest
import blackbox_backprop as bb 

import sys
root_dir = "../"
# sys.path.append(root_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# print('root directory = ')


from src.models import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
from src.models import DYS_Warcraft_Net, Pert_Warcraft_Net, BB_Warcraft_Net
from src.trainer import trainer
from src.utils import edge_to_node, node_to_edge, compute_accuracy

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
        train_dataset = state["train_dataset"]
        val_dataset = state["val_dataset"]
        test_dataset = state["test_dataset"]

        self.grid_size = state["m"]
        A = state["A"].float()
        b = state["b"].float()
        self.num_edges = state["num_edges"]
        self.edge_list = state["edge_list"]
        edge_list_torch = torch.tensor(self.edge_list)

        self.A = A.to(self.device)
        self.b = b.to(self.device)
        self.n_samples = 10
        self.terrain, self.path_edge, self.path_vertex, self.costs = test_dataset[0:self.n_samples]

        self.dys_net = DYS_Warcraft_Net(self.A, self.b, self.edge_list, self.num_edges, self.device)
        self.dys_net.to(self.device)

        self.bb_net = BB_Warcraft_Net(self.edge_list, self.num_edges, self.grid_size, device=self.device)
        self.bb_net.to(self.device)
    
    def test_edge_to_node(self):
        # Test that edge_to_node returns correct node path

        for i in range(self.n_samples):
            path_vertex2 = edge_to_node(self.path_edge[i,:], self.edge_list, self.grid_size, device=self.device)

            self.assertTrue( torch.allclose(path_vertex2.view(-1, self.grid_size, self.grid_size), self.path_vertex[i,:,:].view(-1, self.grid_size, self.grid_size)))

            path_edge2 = node_to_edge(path_vertex2.view(-1, self.grid_size, self.grid_size), self.edge_list, four_neighbors=False)
            self.assertTrue( torch.allclose( path_edge2.view(1, self.num_edges), self.path_edge[i,:].view(1, self.num_edges)) )

            if i%100==0:
                print('i = ', i, ', node_to_edge and edge_to_node passed')

        print('\n\n-------------- edge_to_node and node_to_edge tests passed --------------\n\n')

    def test_A_matrix_and_b_vector(self):

        for i in range(self.A.shape[0]):
            self.assertTrue(torch.sum(self.A[:,i])==0)

        self.assertTrue(torch.all(self.b[1:len(self.b)-1] == 0))
        self.assertTrue(self.b[0]==-1. and self.b[-1]==1.)

        print('\n\n-------------- A matrix and b vector tests passed --------------\n\n')

    def test_compute_accuracy(self):

        accuracy, cost_pred, true_cost = compute_accuracy(self.path_edge, 
                                                          self.path_vertex, 
                                                          self.costs, 
                                                          self.edge_list, 
                                                          self.grid_size, 
                                                          device=self.device, 
                                                          graph_type='E')
        self.assertTrue(accuracy == 1.)
        print('true path (edge) accuracy = ', accuracy, ', cost_pred = ', cost_pred, ', true_cost = ', true_cost)

        accuracy, cost_pred, true_cost = compute_accuracy(self.path_vertex, 
                                                          self.path_vertex, 
                                                          self.costs, 
                                                          self.edge_list, 
                                                          self.grid_size, 
                                                          device=self.device, 
                                                          graph_type='V')
        self.assertTrue(accuracy == 1.)
        print('true path (vertex) accuracy = ', accuracy, ', cost_pred = ', cost_pred, ', true_cost = ', true_cost)

        rand_path_edge = torch.randn(self.path_edge.shape)
        rand_acc, cost_pred, true_cost = compute_accuracy(rand_path_edge, 
                                                          self.path_vertex, 
                                                          self.costs, 
                                                          self.edge_list, 
                                                          self.grid_size, 
                                                          device=self.device, 
                                                          graph_type='E')
        
        print('random path (edge) accuracy = ', rand_acc, ', cost_pred = ', cost_pred, ', true_cost = ', true_cost)
        self.assertTrue(rand_acc < 1.)

        rand_path_vertex = torch.randn(self.path_vertex.shape)
        rand_acc, cost_pred, true_cost = compute_accuracy(rand_path_vertex, 
                                                          self.path_vertex, 
                                                          self.costs, 
                                                          self.edge_list, 
                                                          self.grid_size, 
                                                          device=self.device, 
                                                          graph_type='V')
        
        print('random path (vertex) accuracy = ', rand_acc, ', cost_pred = ', cost_pred, ', true_cost = ', true_cost)
        self.assertTrue(rand_acc < 1.)
        

        print('\n\n-------------- compute_accuracy tests passed --------------\n\n')

    def test_dys_net(self):
        
        path_pred = self.dys_net(self.terrain).detach()
        cost_pred = self.dys_net.data_space_forward(self.terrain).detach()
        
        self.assertTrue(torch.all(cost_pred >= 0))
        print('cost_vec >= 0')
        
        for i in range(path_pred.shape[0]):
            constraint_norm = torch.norm(self.A@path_pred[i,:] - self.b)
            print('for sample ', i, ', |Ax - b| = ', constraint_norm, '  < 1e-1')
            self.assertTrue( constraint_norm <= 1e-1)

        print('\n\n-------------- dys_net tests passed --------------\n\n')

    def test_bb_net(self):
        
        path_pred = self.bb_net(self.terrain)
        self.assertTrue(path_pred.shape == self.path_vertex.shape)

        print('\n\n-------------- bb_net tests passed --------------\n\n')

if __name__ == '__main__':
    unittest.main()