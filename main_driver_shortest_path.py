# Assume path is root directory

from src.utils import create_shortest_path_data
from src.shortest_path.models import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
import matplotlib.pyplot as plt
import time as time
from src.shortest_path.trainer import trainer
import numpy as np
import torch
import os

## Set device
device = 'cuda:0'
print('device: ', device)

## Some fixed hyperparameters
max_epochs = 2
init_lr = 1e-2 # initial learning rate. We're using a scheduler. 
torch.manual_seed(0)

# check that directory to save data exists 
if not os.path.exists('./src/shortest_path/results/'):
    os.makedirs('./src/shortest_path/results/')
if not os.path.exists('./src/shortest_path/saved_weights/'):
    os.makedirs('./src/shortest_path/saved_weights/')

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
grid_size_array = [5,10]

base_data_path = './src/shortest_path/shortest_path_data/'

# -----------------------------------------------------------
# ------------------------ Train DYS ------------------------
# -----------------------------------------------------------

for grid_size in grid_size_array:

  ## Load data
  data_path = base_data_path + 'Shortest_Path_training_data'+str(grid_size)+'.pth'
  state = torch.load(data_path)

  ## Extract data from state
  train_dataset_e = state['train_dataset_e']
  test_dataset_e = state['test_dataset_e']
  train_dataset_v = state['train_dataset_v']
  test_dataset_v = state['test_dataset_v']

  m = state["m"]
  A = state["A"].float()
  b = state["b"].float()
  num_edges = state["num_edges"]
  Edge_list = state["Edge_list"]
  Edge_list_torch = torch.tensor(Edge_list)

  A = A.to(device)
  b = b.to(device)

  ## Load model/network
  DYS_net = ShortestPathNet(A, b, num_vertices = grid_size**2, num_edges = num_edges ,
                    Edges = Edge_list_torch.to(device), context_size = 5, device=device)
  DYS_net.to(device)

  # Train
  print('\n--------------------------------- ----------- TRAINING DYS GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
  start_time = time.time()
  tl_DYS, tt_DYS, ta_DYS = trainer(DYS_net, train_dataset_e, test_dataset_e, grid_size,
                                  max_epochs, init_lr, graph_type='E', Edge_list = Edge_list, max_time=np.inf, device=device)
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
  torch.save(DYS_net.state_dict(), './src/shortest_path/saved_weights/'+'DYS_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './src/shortest_path/results/'+'DYS_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# -----------------------------------------------------------
# ------------------------ Train CVX ------------------------
# -----------------------------------------------------------

for grid_size in grid_size_array:

  ## Load data
  data_path = base_data_path + 'Shortest_Path_training_data'+str(grid_size)+'.pth'
  state = torch.load(data_path)

  ## Extract data from state
  train_dataset_e = state['train_dataset_e']
  test_dataset_e = state['test_dataset_e']
  train_dataset_v = state['train_dataset_v']
  test_dataset_v = state['test_dataset_v']
  m = state["m"]
  A = state["A"].float()
  b = state["b"].float()
  num_edges = state["num_edges"]
  Edge_list = state["Edge_list"]
  Edge_list_torch = torch.tensor(Edge_list)
    
  ## Load model/network
  CVX_net = Cvx_ShortestPathNet(A.float(), b.float(), 5, device=device)
  CVX_net.to(device)

  # Train
  print('\n-------------------------------------------- TRAINING CVX GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
  start_time = time.time()
  tl_CVX, tt_CVX, ta_CVX = trainer(CVX_net, train_dataset_e, test_dataset_e, grid_size,
                          max_epochs, init_lr, graph_type='E', Edge_list = Edge_list, max_time=np.inf, device=device)
  end_time = time.time()
  print('\n time to train CVX GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

  ## Store data
  tl_trained_CVX = tl_CVX
  tt_trained_CVX = tt_CVX
  ta_trained_CVX = ta_CVX

  print('length tl_trained_DYS = ', len(tl_trained_CVX))
  print('length tt_trained_DYS = ', len(tt_trained_CVX))
  print('length ta_trained_DYS = ', len(ta_trained_CVX))

  state = {
            'tl_trained_CVX': tl_trained_CVX,
            'tt_trained_CVX': tt_trained_CVX,
            'ta_trained_CVX': ta_trained_CVX,
            }

  ## Save weights
  torch.save(CVX_net.state_dict(), './src/shortest_path/saved_weights/'+'CVX_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './src/shortest_path/results/'+'CVX_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# ---------------------------------------------------------------
# ------------------------ Train PertOpt ------------------------
# ---------------------------------------------------------------
for grid_size in grid_size_array:

  ## Load data
  data_path = base_data_path + 'Shortest_Path_training_data'+str(grid_size)+'.pth'
  state = torch.load(data_path)

  ## Extract data from state
  train_dataset_e = state['train_dataset_e']
  test_dataset_e = state['test_dataset_e']
  train_dataset_v = state['train_dataset_v']
  test_dataset_v = state['test_dataset_v']
  m = state["m"]
  A = state["A"].float()
  b = state["b"].float()
  num_edges = state["num_edges"]
  Edge_list = state["Edge_list"]
  Edge_list_torch = torch.tensor(Edge_list)
    
  PertOpt_net = Pert_ShortestPathNet(grid_size, context_size=5, device='cpu') # PertOpt not GPU Compatible
  PertOpt_net.to('cpu')

  # Train
  print('\n-------------------------------------------- TRAINING PertOpt GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
  start_time = time.time()
  tl_PertOpt, tt_PertOpt, ta_PertOpt = trainer(PertOpt_net, train_dataset_v,
                                              test_dataset_v, grid_size, max_epochs,
                                              init_lr, graph_type='V', Edge_list = Edge_list,
                                              device='cpu', max_time=np.inf, use_scheduler=False) # note PertOpt is not GPU compatible
  end_time = time.time()
  print('\n time to train PertOpt GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

  ## Store data
  tl_trained_PertOpt = tl_PertOpt
  tt_trained_PertOpt = tt_PertOpt
  ta_trained_PertOpt = ta_PertOpt

  print('length tl_trained_DYS = ', len(tl_trained_PertOpt))
  print('length tt_trained_DYS = ', len(tt_trained_PertOpt))
  print('length ta_trained_DYS = ', len(ta_trained_PertOpt))
  
  state = {
            'tl_trained_PertOpt': tl_trained_PertOpt,
            'tt_trained_PertOpt': tt_trained_PertOpt,
            'ta_trained_PertOpt': ta_trained_PertOpt,
            }

  ## Save weights
  torch.save(PertOpt_net.state_dict(), './src/shortest_path/saved_weights/'+'PertOpt_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './src/shortest_path/results/'+'PertOpt_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')




# ---------------------------------------------------------------
# ------------------------ Train BB ------------------------
# ---------------------------------------------------------------
for grid_size in grid_size_array:

  ## Load data
  data_path = base_data_path + 'Shortest_Path_training_data'+str(grid_size)+'.pth'
  state = torch.load(data_path)

  ## Extract data from state
  train_dataset_e = state['train_dataset_e']
  test_dataset_e = state['test_dataset_e']
  train_dataset_v = state['train_dataset_v']
  test_dataset_v = state['test_dataset_v']
  m = state["m"]
  A = state["A"].float()
  b = state["b"].float()
  num_edges = state["num_edges"]
  Edge_list = state["Edge_list"]
  Edge_list_torch = torch.tensor(Edge_list)
    
  BB_net = BB_ShortestPathNet(grid_size, context_size=5, device=device)
  BB_net.to(device)

  # Train
  print('\n-------------------------------------------- TRAINING BB GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
  start_time = time.time()
  tl_BB, tt_BB, ta_BB = trainer(BB_net, train_dataset_v,
                                              test_dataset_v, grid_size, max_epochs,
                                              init_lr, graph_type='V', Edge_list = Edge_list,
                                              device=device, max_time=np.inf, use_scheduler=False, use_blackbox_backprop=True)
  end_time = time.time()
  print('\n time to train BB GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

  ## Store data
  tl_trained_BB = tl_BB
  tt_trained_BB = tt_BB
  ta_trained_BB = ta_BB

  print('length tl_trained_BB = ', len(tl_trained_BB))
  print('length tt_trained_BB = ', len(tt_trained_BB))
  print('length ta_trained_BB = ', len(ta_trained_BB))
  
  state = {
            'tl_trained_BB': tl_trained_BB,
            'tt_trained_BB': tt_trained_BB,
            'ta_trained_BB': ta_trained_BB,
            }

  ## Save weights
  torch.save(BB_net.state_dict(), './src/shortest_path/saved_weights/'+'BB_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './src/shortest_path/results/'+'BB_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')