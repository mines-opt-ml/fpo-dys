
# from google.colab import drive
# drive.mount('/content/drive')

# pip install cvxpylayers

import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
sys.path.append('./source/') # append files from source folder
# sys.path.append('/content/drive/MyDrive/Projects/2022-SPO-with-DYS')
# sys.path.append('/content/drive/MyDrive/Projects/2022-SPO-with-DYS/source')
from ModelsShortestPath import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
from torch.utils.data import Dataset, TensorDataset, DataLoader
from utils import Edge_to_Node

device = 'cuda:0'

dir = './models/'

"""### Function for computing average KL between two images"""

def compute_average_kl(path_batch, path_pred, Edge_list, m, device, eps=1e-16, is_nodal=False):

  # assumes path_batch and path_pred have shape (n_samples x n_features)

  n_samples = path_batch.shape[0]

  u = Edge_to_Node(path_batch[0], Edge_list, m, device)
  # u_approx = Edge_to_Node(path_pred[0], Edge_list, m, device)

  grid_size = u.shape[0]
  h = 1/grid_size
  kl_val = 0

  for i in range(n_samples):

    u = Edge_to_Node(path_batch[i,:], Edge_list, m, device)
    u = u/torch.sum(u)

    if is_nodal:
      # PertOpt already outputs solutions on nodes (samples x gridsize x gridsize)
      u_approx = path_pred[i,:,:]
    else:
      u_approx = Edge_to_Node(path_pred[i,:], Edge_list, m, device)
    
    u_approx = u_approx/torch.sum(u_approx)

    kl_val = kl_val + h**2 * torch.sum(torch.log(u/(u_approx+eps) + eps) * u) / len(u.view(-1))

  return kl_val/n_samples

"""### Compute KL for all gridsizes"""

# Load DYS Models

KL_array_DYS = []
KL_array_CVX = []
KL_array_PertOpt = []

grid_size_array = [5, 10, 20, 30, 50, 100]

cmap_str = 'viridis'

for grid_size in grid_size_array:

  print('\n\n ------------------------------ GRID SIZE ', str(grid_size), ' ------------------------------')

  ## Load data
  # data_path = '/content/drive/MyDrive/Projects/2022-SPO-with-DYS/shortest_path_data/Shortest_Path_training_data'+str(grid_size)+'.pth'
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

  data_name = 'DYS_' + str(grid_size) + '-by-' + str(grid_size) + '.pth'
  temp_file_name = dir + data_name
  state_dict = torch.load(temp_file_name, map_location=device)

  test_loader = DataLoader(dataset=test_dataset_e, batch_size=200, shuffle=False)
  d_batch, path_batch = next(iter(test_loader))

  d_batch = d_batch.to(device)

  # Load DYS
  DYS_net = ShortestPathNet(A, b, num_vertices = grid_size**2, num_edges = num_edges ,
                      Edges = Edge_list_torch.to(device), context_size = 5)
  DYS_net.to(device)

  DYS_net.load_state_dict(state_dict)

  path_batch =path_batch.to(device)
  path_pred_DYS = DYS_net(d_batch).detach()

  avg_kl_DYS = compute_average_kl(path_batch, path_pred_DYS, Edge_list, m, device)
  print('KL for DYS with gridsize ', str(grid_size), ' = ', avg_kl_DYS)
  KL_array_DYS.append(avg_kl_DYS.cpu())

  data_name = 'PertOpt_' + str(grid_size) + '-by-' + str(grid_size) + '.pth'
  temp_file_name = dir + data_name
  state_dict = torch.load(temp_file_name, map_location=device)

  PertOpt_net = Pert_ShortestPathNet(grid_size, context_size=5, device='cpu')
  PertOpt_net.to('cpu')

  PertOpt_net.load_state_dict(state_dict)

  path_pred_PertOpt = PertOpt_net(d_batch.cpu()).detach()

  avg_kl_PertOpt = compute_average_kl(path_batch.cpu(), path_pred_PertOpt, Edge_list, m, 'cpu', is_nodal=True)
  print('KL for PertOpt with gridsize ', str(grid_size), ' = ', avg_kl_PertOpt)
  KL_array_PertOpt.append(avg_kl_PertOpt.cpu())

  if grid_size<=30:

    # Load CVX
    data_name = 'CVX_' + str(grid_size) + '-by-' + str(grid_size) + '.pth'
    temp_file_name = dir + data_name
    state_dict = torch.load(temp_file_name, map_location=device)

    CVX_net = Cvx_ShortestPathNet(A.float(), b.float(), 5)
    CVX_net.to(device)

    CVX_net.load_state_dict(state_dict)

    path_pred_CVX = CVX_net(d_batch).detach()

    avg_kl_CVX = compute_average_kl(path_batch, path_pred_CVX, Edge_list, m, device)
    print('KL for CVX with gridsize ', str(grid_size), ' = ', avg_kl_CVX)
    KL_array_CVX.append(avg_kl_CVX.cpu())

  # if grid_size<=100:
  #   # Load PertOpt
  #   data_name = 'PertOpt_' + str(grid_size) + '-by-' + str(grid_size) + '.pth'
  #   temp_file_name = dir + data_name
  #   state_dict = torch.load(temp_file_name, map_location=device)

  #   PertOpt_net = Pert_ShortestPathNet(grid_size, context_size=5, device='cpu')
  #   PertOpt_net.to('cpu')

  #   PertOpt_net.load_state_dict(state_dict)

  #   path_pred_PertOpt = PertOpt_net(d_batch.cpu()).detach()

  #   avg_kl_PertOpt = compute_average_kl(path_batch.cpu(), path_pred_PertOpt, Edge_list, m, 'cpu', is_nodal=True)
  #   print('KL for PertOpt with gridsize ', str(grid_size), ' = ', avg_kl_PertOpt)
  #   KL_array_PertOpt.append(avg_kl_PertOpt.cpu())


  # ------------ Plot Predicted Paths -------------# 
  if grid_size <=30 and grid_size >=10:

    # Plot predicted paths gridsize <= 30

    fig1 = plt.figure()
    plt.style.use('default')

    ax = plt.axes()
    for i in range(4):

      ax.imshow(Edge_to_Node(path_pred_DYS[i], Edge_list, m,'cuda:0').cpu(), cmap=cmap_str)
      ax.set_xticks([])
      ax.set_yticks([])
      save_str = './pred_path_plots/pred_path_DYS_grid_'+str(grid_size)+'_img'+str(i)+'.pdf'
      fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
      
      ax.imshow(Edge_to_Node(path_pred_CVX[i], Edge_list, m,'cuda:0').cpu(), cmap=cmap_str)
      ax.set_xticks([])
      ax.set_yticks([])
      save_str = './pred_path_plots/pred_path_CVX_grid_'+str(grid_size)+'_img'+str(i)+'.pdf'
      fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)
     
      ax.imshow(path_pred_PertOpt[i].cpu(), cmap=cmap_str)
      ax.set_xticks([])
      ax.set_yticks([])
      save_str = './pred_path_plots/pred_path_PertOpt_grid_'+str(grid_size)+'_img'+str(i)+'.pdf'
      fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)

      ax.imshow(Edge_to_Node(path_batch[i], Edge_list, m,'cuda:0').cpu(), cmap=cmap_str)
      ax.set_xticks([])
      ax.set_yticks([])
      save_str = './pred_path_plots/true_path_grid_'+str(grid_size)+'_img'+str(i)+'.pdf'
      fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)

  torch.cuda.empty_cache()

"""### Generate KL vs Gridsize Plot"""

title_fontsize = 20
fontsize=20

# PLOT
fig1 = plt.figure()
plt.style.use('seaborn-whitegrid')
ax = plt.axes()
ax.semilogy(grid_size_array, KL_array_DYS, linewidth=2, marker='o', markersize=8);
ax.semilogy(grid_size_array[0:len(KL_array_CVX)], KL_array_CVX, linewidth=2, marker='o', markersize=8);
ax.semilogy(grid_size_array[0:len(KL_array_PertOpt)], KL_array_PertOpt, linewidth=2, marker='o', markersize=8);

ax.set_xlabel("grid size", fontsize=title_fontsize)
ax.legend(['DYS', 'CVX', 'PertOpt'],fontsize=fontsize, loc='upper right')
ax.tick_params(labelsize=fontsize, which='both', direction='in')

save_str = 'KL_vs_gridsize.pdf'
fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)