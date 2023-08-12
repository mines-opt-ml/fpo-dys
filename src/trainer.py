'''
This file implements the training of a diff opt network.

'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, TensorDataset, DataLoader
import time as time
import torch.nn as nn
from src.utils import compute_perfect_path_acc, compute_perfect_path_acc_vertex, edge_to_node, compute_accuracy
import numpy as np

def trainer(net, train_dataset, test_dataset, grid_size, max_epochs,
            learning_rate, graph_type, edge_list, device='cpu', max_time=3600, use_scheduler=True, test_batch_size=200, train_batch_size=200):
    '''
    Train network net using given parameters, for shortest path
    problem on a grid_size-by-grid_size grid graph.
    Training data is automatically loaded.
    type should be either vertex or edge.
    '''

    ## Training setup
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                                  shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = StepLR(optimizer, step_size=max_epochs, gamma=1.0)
    criterion = nn.MSELoss()

    ## Initialize arrays that will be returned.
    test_loss_hist= []
    test_acc_hist = []
    train_time = [0]
    train_loss_ave = 0
    # best_loss = np.inf
    # best_params = net.state_dict()
    
    fmt = '[{:4d}/{:4d}]: train loss = {:7.3e} | test_loss = {:7.3e} ]'

    ## Compute initial loss
    net.eval()
    for d_batch, path_batch in test_loader:
        d_batch = d_batch.to(device)
        path_batch =path_batch.to(device)
        path_pred = net(d_batch).to(device)
        test_loss = criterion(path_batch, path_pred).item()
        test_loss_hist.append(test_loss)
        if graph_type == 'E':
            accuracy = compute_perfect_path_acc(path_pred, path_batch, edge_list, grid_size, device)
            # regret = compute_regret(WW, d_batch, path_batch, path_pred,'E', edge_list, grid_size, device)
        else:
            accuracy = compute_perfect_path_acc_vertex(path_pred, path_batch)
            # regret = compute_regret(WW, d_batch, path_batch, path_pred,'V', edge_list, grid_size, device)
        test_acc_hist.append(accuracy)

    ## Train!
    train_start_time = time.time()
    epoch=0
    epoch_time=0

    # print initial test loss in :7.3e format
    print('initial_test_loss: ', "{:5.2e}".format(test_loss), ' | initial_test_acc: ', "{:<4f}".format(accuracy))

    while epoch <= max_epochs and epoch_time <= max_time:
        for d_batch, path_batch in train_loader:
            d_batch = d_batch.to(device)
            path_batch =path_batch.to(device)
            net.train()
            optimizer.zero_grad()
            path_pred = net(d_batch)
            loss = criterion(path_pred, path_batch)
            train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
            loss.backward()
            optimizer.step()

        # print('epoch:', epoch, ', av. training loss = ', train_loss_ave)
        epoch_time = time.time() - train_start_time
        train_time.append(epoch_time)

        # Evaluate progress on test set. (note one batch is entire dataset)
        net.eval()
        for d_batch, path_batch in test_loader:
            d_batch = d_batch.to(device)
            path_batch =path_batch.to(device)
            path_pred = net(d_batch)    
            test_loss = criterion(path_batch, path_pred).item()
            scheduler.step(test_loss)
            test_loss_hist.append(test_loss)
            # print('epoch: ', epoch, 'test loss is ', test_loss)
            ## Evaluate accuracy
            if graph_type == 'E':
                accuracy = compute_perfect_path_acc(path_pred, path_batch, edge_list, grid_size, device)
                # regret = compute_regret(WW, d_batch, path_batch, path_pred,'E', edge_list, grid_size, device)
            else:
                accuracy = compute_perfect_path_acc_vertex(path_pred, path_batch)
                # regret = compute_regret(WW, d_batch, path_batch, path_pred,'V', edge_list, grid_size, device)
            # print('epoch: ', epoch, 'accuracy is ', accuracy)
            test_acc_hist.append(accuracy)
        
        print('epoch: ', epoch, '| ave_tr_loss: ', "{:5.2e}".format(train_loss_ave), '| te_loss: ', "{:5.2e}".format(test_loss), '| acc.: ', "{:<4f}".format(accuracy), '| lr: ', "{:5.2e}".format(optimizer.param_groups[0]['lr']), '| time: ', "{:<15f}".format(epoch_time))

        epoch += 1
        
        if epoch==max_epochs:
            print('\n ------------------------ \n')
            print('\n Predicted Path \n')
            print('path_pred edge = ', torch.nonzero(path_pred[2,:]))
            print(edge_to_node(path_pred[2,:], edge_list, grid_size, device))
            print('\n True Path \n')
            print(edge_to_node(path_batch[2,:], edge_list, grid_size, device))
            print('path_batch edge = ', torch.nonzero(path_batch[2,:]))
            print('\n ------------------------ \n')


    # return test_loss_hist, train_time, test_acc_hist, best_params
    return test_loss_hist, train_time, test_acc_hist