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
            accuracy = compute_perfect_path_acc(path_pred, path_batch)
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
                accuracy = compute_perfect_path_acc(path_pred, path_batch)
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

#------------------------------------------------------------------------------------------------------------------------------------------------
# Warcraft Trainer
#------------------------------------------------------------------------------------------------------------------------------------------------

def trainer_warcraft(net, train_dataset, val_dataset, test_dataset, 
                     grid_size, max_epochs, learning_rate, edge_list, 
                     device='cpu', use_scheduler=True, 
                     test_batch_size=200, train_batch_size=200):
    '''
    Train network net using given parameters, for warcraft shortest path problem on a grid_size-by-grid_size grid graph.
    Training data is automatically loaded in both vertex and edge form.
    Type should be either vertex or edge.
    '''

    ## Training setup
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                                  shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=test_batch_size, 
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = StepLR(optimizer, step_size=max_epochs, gamma=1.0)
    criterion = nn.MSELoss()

    ## Initialize arrays that will be returned.
    val_loss_hist= []
    val_acc_hist = []
    val_cost_pred_hist = []
    train_time = [0]
    train_loss_ave = 0

    best_acc = 0.0
    best_params = net.state_dict()
    
    fmt = '[{:4d}/{:4d}]: train loss = {:5.2e} | val_loss = {:5.2e}]'

    ## Compute initial loss
    net.eval()
    for terrain_batch, path_batch_edge, path_batch_vertex, costs_batch in val_loader:

        terrain_batch = terrain_batch.to(device)
        path_batch_edge = path_batch_edge.to(device)
        path_batch_vertex = path_batch_vertex.to(device)
        costs_batch = costs_batch.to(device)

        path_pred_edge = net(terrain_batch)

        val_loss = criterion(path_batch_edge, path_pred_edge).item()
        val_loss_hist.append(val_loss)

        # compute accuracy based on optimal cost. 
        # pred_batch_edge_form=True means path_pred_edge is in edge form.
        val_acc, val_cost_pred, val_cost_true = compute_accuracy(path_pred_edge, path_batch_vertex, costs_batch, edge_list, grid_size, device, pred_batch_edge_form=True)
        val_acc_hist.append(val_acc)
        val_cost_pred_hist.append(val_cost_pred)

    for terrain_batch, path_batch_edge, path_batch_vertex, costs_batch in test_loader:

        terrain_batch = terrain_batch.to(device)
        path_batch_edge = path_batch_edge.to(device)
        path_batch_vertex = path_batch_vertex.to(device)
        costs_batch = costs_batch.to(device)

        path_pred_edge = net(terrain_batch)

        test_loss = criterion(path_batch_edge, path_pred_edge).item()
        # val_loss_hist.append(val_loss)

        # compute accuracy based on optimal cost. 
        # pred_batch_edge_form=True means path_pred_edge is in edge form.
        test_acc, test_cost_pred, test_cost_true = compute_accuracy(path_pred_edge, path_batch_vertex, costs_batch, edge_list, grid_size, device, pred_batch_edge_form=True)
        
    ## Train!
    train_start_time = time.time()
    epoch=0
    epoch_time=0

    # print initial test loss in :7.3e format
    print('INITIAL VALUES:')
    print('val_loss: ', "{:5.2e}".format(val_loss), 
          ' | val_acc: ', "{:<4.3f}".format(val_acc), 
          ' | val_cost_pred: ', "{:5.2e}".format(val_cost_pred),
          ' | true val_cost: ', "{:5.2e}".format(val_cost_true),
          'test_loss: ', "{:5.2e}".format(test_loss), 
          ' | test_acc: ', "{:<4.3f}".format(test_acc),
          ' | test_cost_pred: ', "{:5.2e}".format(test_cost_pred),
          ' | true test_cost: ', "{:5.2e}".format(test_cost_true))

    while epoch <= max_epochs:
        
        # training step
        for terrain_batch, path_batch_edge, _, _ in train_loader:

            terrain_batch = terrain_batch.to(device)
            path_batch_edge =path_batch_edge.to(device)
            net.train()
            optimizer.zero_grad()
            path_pred = net(terrain_batch)
            loss = criterion(path_pred, path_batch_edge)
            train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - train_start_time
        train_time.append(epoch_time)

        # Evaluate progress on val set. (note one batch is entire dataset)
        net.eval()
        for terrain_batch, path_batch_edge, path_batch_vertex, costs_batch in val_loader:

            terrain_batch = terrain_batch.to(device)
            path_batch_edge = path_batch_edge.to(device)
            path_batch_vertex = path_batch_vertex.to(device)
            costs_batch = costs_batch.to(device)

            path_pred_edge = net(terrain_batch)

            val_loss = criterion(path_batch_edge, path_pred_edge).item()
            val_loss_hist.append(val_loss)

            # compute accuracy based on optimal cost. 
            # pred_batch_edge_form=True means path_pred_edge is in edge form.
            val_acc, val_cost_pred, val_cost_true = compute_accuracy(path_pred_edge, path_batch_vertex, costs_batch, edge_list, grid_size, device, pred_batch_edge_form=True)
            val_acc_hist.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = net.state_dict().copy()
        
        print('epoch: ', epoch, '| ave_tr_loss: ', "{:5.2e}".format(train_loss_ave), 
              '| val_loss: ', "{:5.2e}".format(val_loss), 
              '| val_acc.: ', "{:<4.3f}".format(val_acc), 
              '| val_cost_pred: ', "{:5.2e}".format(val_cost_pred),
              '| val_cost_true:', "{:5.2e}".format(val_cost_true),
              '| lr: ', "{:5.2e}".format(optimizer.param_groups[0]['lr']), 
              '| time: ', "{:<15f}".format(epoch_time))

        epoch += 1
        
        if epoch==max_epochs:
            print('\n ------------------------ \n')
            print('\n Predicted Path \n')
            print('path_pred edge = ', torch.nonzero(path_pred[2,:]))
            print(edge_to_node(path_pred[2,:], edge_list, grid_size, device))
            print('\n True Path \n')
            print(edge_to_node(path_batch_edge[2,:], edge_list, grid_size, device))
            print('path_batch edge = ', torch.nonzero(path_batch_edge[2,:]))
            print('\n ------------------------ \n')


    # return test_loss_hist, train_time, test_acc_hist, best_params
    
    # compute test loss using best params
    net.load_state_dict(best_params)
    net.eval()
    for terrain_batch, path_batch_edge, path_batch_vertex, costs_batch in test_loader:
            
            terrain_batch = terrain_batch.to(device)
            path_batch_edge = path_batch_edge.to(device)
            path_batch_vertex = path_batch_vertex.to(device)
            costs_batch = costs_batch.to(device)
    
            path_pred_edge = net(terrain_batch)
    
            test_loss = criterion(path_batch_edge, path_pred_edge).item()
    
            # compute accuracy based on optimal cost. 
            # pred_batch_edge_form=True means path_pred_edge is in edge form.
            test_acc, test_cost_pred, test_cost_true = compute_accuracy(path_pred_edge, path_batch_vertex, costs_batch, edge_list, grid_size, device, pred_batch_edge_form=True)

    print('final test loss is ', "{:5.2e}".format(test_loss), ' | final test acc. is ', "{:<4.3f}".format(test_acc), ' | final test cost pred is ', "{:5.2e}".format(test_cost_pred), ' | final test cost true is ', "{:5.2e}".format(test_cost_true))

    return best_params, val_loss_hist, val_acc_hist, test_loss, test_acc, train_time
    