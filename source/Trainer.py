'''
This file implements the training of a diff opt network.

'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
import time as time
import torch.nn as nn
from utils import Edge_to_Node, Compute_Perfect_Path_Acc, Compute_Perfect_Path_Acc_V

def trainer(net, train_dataset, test_dataset, grid_size, max_epochs,
            learning_rate, graph_type, Edge_list, device='cuda:0', max_time=3600):
    '''
    Train network net using given parameters, for shortest path
    problem on a grid_size-by-grid_size grid graph.
    Training data is automatically loaded.
    type should be either vertex or edge.
    '''

    ## Training setup
    test_size = 200
    train_loader = DataLoader(dataset=train_dataset, batch_size=200,
                                  shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.MSELoss()

    ## Initialize arrays that will be returned.
    test_loss_hist= []
    test_acc_hist = []
    train_time = [0]
    train_loss_ave = 0
    
    fmt = '[{:4d}/{:4d}]: train loss = {:7.3e} | test_loss = {:7.3e} ]'

    ## Compute initial loss
    net.eval()
    for d_batch, path_batch in test_loader:
        d_batch = d_batch.to(device)
        path_batch =path_batch.to(device)
        path_pred = net(d_batch)
        test_loss = criterion(path_batch, path_pred).item()
        test_loss_hist.append(test_loss)

    ## Train!
    train_start_time = time.time()
    epoch=0
    epoch_time=0

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

        print('epoch:', epoch, ', av. training loss = ', train_loss_ave)
        epoch_time = time.time() - train_start_time
        train_time.append(epoch_time)

        # Evaluate progress on test set.
        net.eval()
        for d_batch, path_batch in test_loader:
            d_batch = d_batch.to(device)
            path_batch =path_batch.to(device)
            path_pred = net(d_batch)
            test_loss = criterion(path_batch, path_pred).item()
            scheduler.step(test_loss)
            test_loss_hist.append(test_loss)
            print('epoch: ', epoch, 'test loss is ', test_loss)
            ## Evaluate accuracy
            if graph_type == 'E':
                accuracy = Compute_Perfect_Path_Acc(path_pred, path_batch, Edge_list, grid_size, device)
            else:
                accuracy = Compute_Perfect_Path_Acc_V(path_pred, path_batch)
            print('epoch: ', epoch, 'accuracy is ', accuracy)
            test_acc_hist.append(accuracy)

        epoch += 1



    return test_loss_hist, train_time, test_acc_hist