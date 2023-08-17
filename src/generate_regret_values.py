import numpy as np
import matplotlib.pyplot as plt

import torch

import sys
sys.path.append('./src/') # append files from source folder
from models import ShortestPathNet, Cvx_ShortestPathNet, Pert_ShortestPathNet, BB_ShortestPathNet
from torch.utils.data import Dataset, TensorDataset, DataLoader
from utils import Edge_to_Node, compute_regret
import time as time

device='cuda:0'



"""### Load trained models"""

dir = './models/'

regret_array_DYS = []
regret_array_CVX = []
regret_array_PertOpt = []

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
    WW = state["WW"].float()
    num_edges = state["num_edges"]
    Edge_list = state["Edge_list"]
    Edge_list_torch = torch.tensor(Edge_list)

    # ------------------------------------------------------------------------------------------------
    # Compute Regret for DYS
    # ------------------------------------------------------------------------------------------------

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

    path_batch = path_batch.to(device)
    path_pred_DYS = DYS_net(d_batch).detach()

    # ------------------------------------------------------------------------------------------------
    # Compute Regret for PertOpt
    # ------------------------------------------------------------------------------------------------
    start_time = time.time()
    regret_DYS = compute_regret(WW, d_batch, path_batch, path_pred_DYS,'E', Edge_list, grid_size, device)
    print('regret DYS = ', regret_DYS)
    regret_array_DYS.append(regret_DYS.cpu())
    end_time = time.time()
    regret_time = end_time - start_time
    print('regret time = ', regret_time)

    data_name = 'PertOpt_' + str(grid_size) + '-by-' + str(grid_size) + '.pth'
    temp_file_name = dir + data_name
    state_dict = torch.load(temp_file_name, map_location=device)

    PertOpt_net = Pert_ShortestPathNet(grid_size, context_size=5, device='cpu')
    PertOpt_net.to('cpu')

    PertOpt_net.load_state_dict(state_dict)

    path_pred_PertOpt = PertOpt_net(d_batch.cpu()).detach()

    start_time = time.time()
    regret_PertOpt = compute_regret(WW, d_batch, path_batch, path_pred_PertOpt.to(device),'NotE', Edge_list, grid_size, device)
    print('regret PertOpt = ', regret_PertOpt)
    regret_array_PertOpt.append(regret_PertOpt.cpu())
    end_time = time.time()
    regret_time = end_time - start_time
    print('regret time = ', regret_time)


    if grid_size<=30:
        # Load CVX
        data_name = 'CVX_' + str(grid_size) + '-by-' + str(grid_size) + '.pth'
        temp_file_name = dir + data_name
        state_dict = torch.load(temp_file_name, map_location=device)

        CVX_net = Cvx_ShortestPathNet(A.float(), b.float(), 5)
        CVX_net.to(device)

        CVX_net.load_state_dict(state_dict)

        path_pred_CVX = CVX_net(d_batch).detach()

        start_time = time.time()
        regret_CVX = compute_regret(WW, d_batch, path_batch, path_pred_CVX,'E', Edge_list, grid_size, device)
        print('regret CVX = ', regret_CVX)
        end_time = time.time()
        regret_time = end_time - start_time
        print('regret time = ', regret_time)

        regret_array_CVX.append(regret_CVX.cpu())


"""### Generate Regret vs Gridsize Plot"""

title_fontsize = 20
fontsize=20

# PLOT
fig1 = plt.figure()
plt.style.use('seaborn-whitegrid')
ax = plt.axes()
ax.semilogy(grid_size_array, regret_array_DYS, linewidth=2, marker='o', markersize=8);
ax.semilogy(grid_size_array[0:len(regret_array_CVX)], regret_array_CVX, linewidth=2, marker='o', markersize=8);
ax.semilogy(grid_size_array[0:len(regret_array_PertOpt)], regret_array_PertOpt, linewidth=2, marker='o', markersize=8);

ax.set_xlabel("grid size", fontsize=title_fontsize)
ax.legend(['DYS', 'CVX', 'PertOpt'],fontsize=fontsize, loc='upper right')
ax.tick_params(labelsize=fontsize, which='both', direction='in')

save_str = 'regret_vs_gridsize.pdf'
fig1.savefig(save_str, dpi=300 , bbox_inches="tight", pad_inches=0.0)



