# Assume path is root directory

from src.models import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
from src.models import DYS_Warcraft_Net, Pert_Warcraft_Net
import matplotlib.pyplot as plt
import time as time
from src.trainer import trainer_warcraft
import numpy as np
import torch
import os

## Set device
device = 'cuda:2'
print('device: ', device)

## Some fixed hyperparameters
max_epochs = 100
init_lr = 1e-5 # initial learning rate. We're using a scheduler. 
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
base_data_path = './src/warcraft/warcraft_data/'

train_batch_size = 1012
test_batch_size = 1000

# # -----------------------------------------------------------
# # ------------------------ Train DYS ------------------------
# # -----------------------------------------------------------

# ## Load data
# data_path = base_data_path + 'Warcraft_training_data'+str(grid_size)+'.pth'
# state = torch.load(data_path)

# ## Extract data from state
# train_dataset = state['train_dataset']
# val_dataset = state['val_dataset']
# test_dataset = state['test_dataset']


# m= state["m"]
# A = state["A"].float()
# b = state["b"].float()
# num_edges = state["num_edges"]
# edge_list = state["edge_list"]
# edge_list_torch = torch.tensor(edge_list)

# A = A.to(device)
# b = b.to(device)

# ## Load model/network
# DYS_net = DYS_Warcraft_Net(A, b, edge_list, num_edges=num_edges, device=device)
# DYS_net.to(device)

# # Train
# print('\n-------------------------------------------- TRAINING DYS Warcraft Grid ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
# start_time = time.time()
# best_params_DYS, val_loss_hist_DYS, val_acc_hist_DYS, test_loss_DYS, test_acc_DYS, train_time_DYS = trainer_warcraft(DYS_net, train_dataset, val_dataset, test_dataset, 
#                                  grid_size, max_epochs, init_lr, edge_list, 
#                                  use_scheduler=False, device=device, 
#                                  train_batch_size=train_batch_size, 
#                                  test_batch_size=test_batch_size)

# end_time = time.time()
# print('\n time to train DYS GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

# ## Store data
# # best_params_DYS
# # tl_trained_DYS = test_loss_DYS
# # tt_trained_DYS = train_time_DYS
# # ta_trained_DYS = test_acc_DYS

# # print('length tl_trained_DYS = ', len(tl_trained_DYS))
# # print('length tt_trained_DYS = ', len(tt_trained_DYS))
# # print('length ta_trained_DYS = ', len(ta_trained_DYS))

# state = {
#         'val_loss_hist_DYS': val_loss_hist_DYS,
#         'val_acc_hist_DYS': val_acc_hist_DYS,
#         'test_loss_DYS': test_loss_DYS,
#         'test_acc_DYS': test_acc_DYS,
#         'train_time_DYS': train_time_DYS
#         }

# # Save weights
# torch.save(best_params_DYS, './src/warcraft/saved_weights/'+'DYS_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# ## Save Histories
# torch.save(state, './src/warcraft/results/'+'DYS_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# ---------------------------------------------------------------
# ------------------------ Train PertOpt ------------------------
# ---------------------------------------------------------------

## Load data
data_path = base_data_path + 'Warcraft_training_data'+str(grid_size)+'.pth'
state = torch.load(data_path)

## Extract data from state
train_dataset = state['train_dataset']
val_dataset = state['val_dataset']
test_dataset = state['test_dataset']


m= state["m"]
# A = state["A"].float()
# b = state["b"].float()
num_edges = state["num_edges"]
edge_list = state["edge_list"]
edge_list_torch = torch.tensor(edge_list)

# A = A.to(device)
# b = b.to(device)


PertOpt_net = Pert_Warcraft_Net(edges=edge_list, num_edges=num_edges, m=m, device='cpu')
PertOpt_net.to('cpu')

# # Train
print('\n-------------------------------------------- TRAINING PertOpt Warcraft GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')

start_time = time.time()
best_params_PertOpt, val_loss_hist_PertOpt, val_acc_hist_PertOpt, test_loss_PertOpt, test_acc_PertOpt, train_time_PertOpt = trainer_warcraft(PertOpt_net, train_dataset, val_dataset, test_dataset, 
                                 grid_size, max_epochs, init_lr, edge_list, 
                                 use_scheduler=False, device='cpu', 
                                 train_batch_size=train_batch_size, 
                                 test_batch_size=test_batch_size)
end_time = time.time()
print('\n time to train PertOpt GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

state = {
        'val_loss_hist_PertOpt': val_loss_hist_PertOpt,
        'val_acc_hist_PertOpt': val_acc_hist_PertOpt,
        'test_loss_PertOpt': test_loss_PertOpt,
        'test_acc_PertOpt': test_acc_PertOpt,
        'train_time_PertOpt': train_time_PertOpt
        }

# Save weights
torch.save(best_params_PertOpt, './src/warcraft/saved_weights/'+'PertOpt_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

## Save Histories
torch.save(state, './src/warcraft/results/'+'PertOpt_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')