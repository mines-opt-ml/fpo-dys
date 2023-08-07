# Assume path is root directory

from src.models import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
from src.models import DYS_Warcraft_Net
import matplotlib.pyplot as plt
import time as time
from src.trainer import trainer
import numpy as np
import torch
import os

## Set device
device = 'cuda:0'
print('device: ', device)

## Some fixed hyperparameters
max_epochs = 20
init_lr = 1e-2 # initial learning rate. We're using a scheduler. 
torch.manual_seed(0)

# check that directory to save data exists 
if not os.path.exists('./src/warcraft/results/'):
    os.makedirs('./src/warcraft/results/')
if not os.path.exists('./src/warcraft/saved_weights/'):
    os.makedirs('./src/warcraft/saved_weights/')

## Some arrays
tl_trained_DYS = [] # test loss
tt_trained_DYS = [] # training time
ta_trained_DYS = [] # test accuracy of trained model
ne_trained_DYS = [] # number of epochs completed during training

tl_trained_CVX = []
tt_trained_CVX = []
ta_trained_CVX = []
ne_trained_CVX = []

tl_trained_PertOpt = []
tt_trained_PertOpt = []
ta_trained_PertOpt = []
ne_trained_PertOpt = []

tl_trained_BB = []
tt_trained_BB = []
ta_trained_BB = []
ne_trained_BB = []

# Define Grid array for all models to solve
# grid_size_array = [5,10,20,30, 50, 100]
# grid_size_array = [5,10]
grid_size = 12
base_data_path = './source/warcraft/warcraft_data/'

# -----------------------------------------------------------
# ------------------------ Train DYS ------------------------
# -----------------------------------------------------------

## Load data
data_path = base_data_path + 'Warcraft_training_data'+str(grid_size)+'.pth'
state = torch.load(data_path)

## Extract data from state
train_dataset_e = state['train_dataset_e']
test_dataset_e = state['test_dataset_e']
train_dataset_v = state['train_dataset_v']
test_dataset_v = state['test_dataset_v']

m= state["m"]
A = state["A"].float()
b = state["b"].float()
num_edges = state["num_edges"]
edge_list = state["edge_list"]
edge_list_torch = torch.tensor(edge_list)

A = A.to(device)
b = b.to(device)

## Load model/network
DYS_net = DYS_Warcraft_Net(A, b, edge_list, num_edges=num_edges, device=device)
DYS_net.to(device)

# Train
print('\n--------------------------------- ----------- TRAINING DYS Warcraft Grid ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
start_time = time.time()
tl_DYS, tt_DYS, ta_DYS = trainer(DYS_net, train_dataset_e, test_dataset_e, grid_size,
                                max_epochs, init_lr, graph_type='E', edge_list = edge_list, max_time=np.inf, device=device, train_batch_size=256, test_batch_size=256)
end_time = time.time()
print('\n time to train DYS GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

## Store data
tl_trained_DYS = tl_DYS
tt_trained_DYS = tt_DYS
ta_trained_DYS = ta_DYS

print('length tl_trained_DYS = ', len(tl_trained_DYS))
print('length tt_trained_DYS = ', len(tt_trained_DYS))
print('length ta_trained_DYS = ', len(ta_trained_DYS))

state = {
        'tl_trained_DYS': tl_trained_DYS,
        'tt_trained_DYS': tt_trained_DYS,
        'ta_trained_DYS': ta_trained_DYS,
        }

# Save weights
torch.save(DYS_net.state_dict(), './src/warcraft/saved_weights/'+'DYS_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

## Save Histories
torch.save(state, './src/warcraft/results/'+'DYS_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# # ---------------------------------------------------------------
# # ------------------------ Train PertOpt ------------------------
# # ---------------------------------------------------------------

# ## Load data
# data_path = base_data_path + 'Warcraft_training_data'+str(grid_size)+'.pth'
# state = torch.load(data_path)

# ## Extract data from state
# train_dataset_e = state['train_dataset_e']
# test_dataset_e = state['test_dataset_e']
# train_dataset_v = state['train_dataset_v']
# test_dataset_v = state['test_dataset_v']
# m = state["m"]
# A = state["A"].float()
# b = state["b"].float()
# num_edges = state["num_edges"]
# edge_list = state["Edge_list"]
# edge_list_torch = torch.tensor(edge_list)

# PertOpt_net = Pert_Warcraft_Net(num_edges=num_edges, device=device)
# PertOpt_net.to('cpu')

# # Train
# print('\n-------------------------------------------- TRAINING PertOpt GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
# start_time = time.time()
# tl_PertOpt, tt_PertOpt, ta_PertOpt = trainer(PertOpt_net, train_dataset_v,
#                                             test_dataset_v, grid_size, max_epochs,
#                                             init_lr, graph_type='V', edge_list = edge_list,
#                                             device='cpu', max_time=np.inf, use_scheduler=False) # note PertOpt is not GPU compatible
# end_time = time.time()
# print('\n time to train PertOpt GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

# ## Store data
# tl_trained_PertOpt = tl_PertOpt
# tt_trained_PertOpt = tt_PertOpt
# ta_trained_PertOpt = ta_PertOpt

# print('length tl_trained_PertOpt = ', len(tl_trained_PertOpt))
# print('length tt_trained_PertOpt = ', len(tt_trained_PertOpt))
# print('length ta_trained_PertOpt = ', len(ta_trained_PertOpt))

# state = {
#         'tl_trained_PertOpt': tl_trained_PertOpt,
#         'tt_trained_PertOpt': tt_trained_PertOpt,
#         'ta_trained_PertOpt': ta_trained_PertOpt,
#         }

# ## Save weights
# torch.save(PertOpt_net.state_dict(), './src/warcraft/saved_weights/'+'PertOpt_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# ## Save Histories
# torch.save(state, './src/warcraft/results/'+'PertOpt_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')