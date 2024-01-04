import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import time as time
import torch.nn as nn
import pyepo
import os

def trainer(net, train_dataset, test_dataset, val_dataset, grid_size, learning_rate, model_type, device='mps'):

    ## Training setup
    batch_size = 256
    loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True)
    loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False)
    loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # TODO: consider using lr scheduler

    # Initialize loss and evaluation metric
    criterion = nn.MSELoss()
    metric = pyepo.metric.regret

    if model_type == "BBOpt":
        dbb = pyepo.func.blackboxOpt(net.shortest_path_solver, lambd=20, processes=1)
    elif model_type == "PertOpt":
        ptb = pyepo.func.perturbedOpt(net.shortest_path_solver, n_samples=3, sigma=1.0, processes=2)
    elif model_type == "DYS" or model_type == "CVX":
        pass
    else:
        raise TypeError("Please choose a supported model!")

    ## Initialize arrays that will be returned and checkpoint directory
    val_loss_hist= []
    epoch_time_hist = []
    max_time = 1200
    checkpt_path = './src/shortest_path/saved_weights/' + model_type + '/'
    if not os.path.exists(checkpt_path):
        os.makedirs(checkpt_path)

    # TODO
    # The PyEPO regret evaluator isn't very good at handling batches.
    # Probably better off writing a custom regret evaluator.
    net.eval()
    net.to('cpu')
    best_val_loss = metric(net, net.shortest_path_solver, loader_val)

    print('Initial validation loss is ', best_val_loss)
    val_loss_hist.append(best_val_loss)
    time_till_best_val_loss = 0

     ## Compute initial test loss
    best_test_loss = metric(net,net.shortest_path_solver, loader_test)
   
    ## Train!
    epoch=1
    train_time=0
    train_loss_ave = 0