# Assume path is currently 2022-SPO-with-DYS


import sys
sys.path.append('./source/') # append files from source folder

import torch
from GenerateShortestPathData import create_shortest_path_data
from ModelsShortestPath import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
import matplotlib.pyplot as plt
import time as time
from Trainer import trainer
import numpy as np

## Set device
device = 'cuda:0'

## Some fixed hyperparameters
max_epochs = 200
init_lr = 1e-2 # initial learning rate. We're using a scheduler.
torch.manual_seed(0)

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
grid_size_array = [5,10,15,20,25,30,50]

# -----------------------------------------------------------
# ------------------------ Train DYS ------------------------
# -----------------------------------------------------------

for grid_size in grid_size_array:

  ## Load data
  data_path = './shortest_path_data/Shortest_Path_training_data'+str(grid_size)+'.pth'
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
  DYS_net = ShortestPathNet(A, b, num_vertices = grid_size**2, num_edges = num_edges ,
                    Edges = Edge_list_torch.to(device), context_size = 5)
  DYS_net.to(device)

  # Train
  print('\n-------------------------------------------- TRAINING DYS GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
  start_time = time.time()
  tl_DYS, tt_DYS, ta_DYS = trainer(DYS_net, train_dataset_e, test_dataset_e, grid_size,
                                  max_epochs, init_lr, graph_type='E', Edge_list = Edge_list, max_time=np.inf)
  end_time = time.time()
  print('\n time to train DYS GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

  ## Store data
  tl_trained_DYS.append(tl_DYS[-1])
  tt_trained_DYS.append(tt_DYS[-1])
  ta_trained_DYS.append(ta_DYS[-1])
  ne_trained_DYS.append(len(tt_DYS))

  state = {
            'tl_trained_DYS': tl_trained_DYS,
            'tt_trained_DYS': tl_trained_DYS,
            'ta_trained_DYS': ta_trained_DYS,
            'ne_trained_DYS': ne_trained_DYS,
            }

  # Save weights
  torch.save(DYS_net.state_dict(), './models/'+'DYS_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './results/'+'DYS_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# -----------------------------------------------------------
# ------------------------ Train CVX ------------------------
# -----------------------------------------------------------

for grid_size in grid_size_array:

  ## Load data
  data_path = './shortest_path_data/Shortest_Path_training_data'+str(grid_size)+'.pth'
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
  CVX_net = Cvx_ShortestPathNet(A.float(), b.float(), 5)
  CVX_net.to(device)

  # Train
  print('\n-------------------------------------------- TRAINING CVX GRID ' + str(grid_size) + '-by-' + str(grid_size) + ' --------------------------------------------')
  start_time = time.time()
  try:
      tl_CVX, tt_CVX, ta_CVX = trainer(CVX_net, train_dataset_e, test_dataset_e, grid_size,
                            max_epochs, init_lr, graph_type='E', Edge_list = Edge_list, max_time=np.inf)
  except:
      print('CVX-Net failed to train!')
  end_time = time.time()
  print('\n time to train CVX GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

  ## Store data
  tl_trained_CVX.append(tl_CVX[-1])
  tt_trained_CVX.append(tt_CVX[-1])
  ta_trained_CVX.append(ta_CVX[-1])
  ne_trained_CVX.append(len(tt_CVX))

  state = {
            'tl_trained_CVX': tl_trained_CVX,
            'tt_trained_CVX': tl_trained_CVX,
            'ta_trained_CVX': ta_trained_CVX,
            'ne_trained_CVX': ne_trained_CVX,
            }

  ## Save weights
  torch.save(CVX_net.state_dict(), './models/'+'CVX_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './results/'+'CVX_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

# -----------------------------------------------------------
# ------------------------ Train PertOpt ------------------------
# -----------------------------------------------------------

for grid_size in grid_size_array:

  ## Load data
  data_path = './shortest_path_data/Shortest_Path_training_data'+str(grid_size)+'.pth'
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
                                              device='cpu') # note PertOpt is not GPU compatible
  end_time = time.time()
  print('\n time to train PertOpt GRID ' + str(grid_size) + '-by-' + str(grid_size), ' = ', end_time-start_time, ' seconds')

  ## Store data
  tl_trained_PertOpt.append(tl_PertOpt[-1])
  tl_trained_PertOpt.append(tt_PertOpt[-1])
  ta_trained_PertOpt.append(ta_PertOpt[-1])
  ne_trained_PertOpt.append(len(tt_PertOpt))
  
  state = {
            'tl_trained_PertOpt': tl_trained_PertOpt,
            'tt_trained_PertOpt': tl_trained_PertOpt,
            'ta_trained_PertOpt': ta_trained_PertOpt,
            'ne_trained_PertOpt': ne_trained_PertOpt,
            }

  ## Save weights
  torch.save(PertOpt_net.state_dict(), './models/'+'PertOpt_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')

  ## Save Histories
  torch.save(state, './results/'+'PertOpt_results_'+str(grid_size) + '-by-' + str(grid_size) + '.pth')


# plt.plot(tt_DYS, [item/tl_DYS[0] for item in tl_DYS], label="DYS",
#              color = 'blue', linestyle= '--')

# plt.legend()
# plt.title('Normalized Test loss (y-axis) vs Training time elapsed')
# plt.xlabel('time in s')
# plt.ylabel('Normalized test loss')
# plt.show()

# plt.plot(tl_DYS, label="DYS", color = 'blue', linestyle= '--')

# plt.plot(tl_CVX, label="CVX", color = 'red', linestyle= '--')

# plt.plot(tl_PertOpt, label="PertOpt", color = 'black', linestyle= '--')
# plt.legend()
# plt.title('Test loss (y-axis) vs epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Test loss')

# plt.plot(ta_DYS, label="DYS", color = 'blue', linestyle= '--')

# plt.plot(ta_CVX, label="CVX", color = 'red', linestyle= '--')

# plt.plot(ta_PertOpt, label="PertOpt", color = 'black', linestyle= '--')
# plt.legend()
# plt.title('Test loss (y-axis) vs epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Test accuracy')

# # Plots entire thing.

# # sizes = range(5,31,5)
# sizes = grid_size_array
# plt.plot(sizes, ta_trained_DYS, label="DYS", color = 'blue', linestyle= '--')

# plt.plot(sizes, ta_trained_CVX, label="CVX", color = 'red', linestyle= '--')

# plt.plot(sizes, ta_trained_PertOpt, label="PertOpt", color = 'black', linestyle= '--')
# plt.legend()
# plt.title('Test Acc (y-axis) vs epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Test accuracy')

