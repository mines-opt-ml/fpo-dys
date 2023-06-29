'''
This file implements the training of a diff opt network.

'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
import time as time
import torch.nn as nn
from knapsack_utils import RegretLoss, Compute_Test_Loss
import pyepo

def trainer(net, train_dataset, test_dataset, val_dataset, num_item, num_knapsack, max_epochs,
            learning_rate, model_type, device='cuda:0'):
    '''
    Train network net using given parameters, for shortest path
    problem on a grid_size-by-grid_size grid graph.
    Training data is automatically loaded.
    type should be either vertex or edge.
    '''

    ## Training setup
    batch_size = 200 # 256
    loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True)
    loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False)
    loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 5e-4)
    if model_type == "DYS":
        criterion = RegretLoss(num_item, num_knapsack, device=device)
    elif model_type == "Two-stage":
        criterion = nn.MSELoss()
        regret = RegretLoss(num_item, num_knapsack, device=device)
    elif model_type == "SPO+":
        criterion = pyepo.func.SPOPlus(net.knapsack_solver, processes=1)
        regret = RegretLoss(num_item, num_knapsack, device=device)
    elif model_type == "BBOpt":
        dbb = pyepo.func.blackboxOpt(net.knapsack_solver, lambd=20, processes=1)
        criterion = RegretLoss(num_item, num_knapsack, device=device)
    elif model_type == "PertOpt":
        ptb = pyepo.func.perturbedOpt(net.knapsack_solver, n_samples=10, sigma=0.5, processes=1)
        criterion = RegretLoss(num_item, num_knapsack, device=device)
    elif model_type == "PertOpt-FY":
        criterion = pyepo.func.perturbedFenchelYoung(net.knapsack_solver, n_samples=10, sigma=0.5, processes=1)
        regret = RegretLoss(num_item, num_knapsack, device=device)
    else:
        raise TypeError("Please choose a supported model!")
    
        
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    ## Initialize arrays that will be returned and checkpoint directory
    val_loss_hist= []
    epoch_time_hist = []
    max_time = 3600
    checkpt_path = './models/' + model_type + '/' 

     ## Compute initial validation loss
    if model_type == "DYS" or model_type == "BBOpt" or model_type == "PertOpt":
        metric = criterion
    else:
        metric = regret

    best_val_loss = Compute_Test_Loss(net,loader_val, model_type, metric, num_knapsack, num_item, device)

    print('Initial validation loss is ', best_val_loss)
    val_loss_hist.append(best_val_loss)
    time_till_best_val_loss = 0

     ## Compute initial test loss
    best_test_loss = Compute_Test_Loss(net,loader_test, model_type, metric, num_knapsack, num_item, device)
    # net.eval()
    # test_loss = 0
    # for d_batch, w_batch, opt_sol, opt_value in loader_test:
    #     d_batch = d_batch.to(device)
    #     w_batch = w_batch.to(device)
    #     opt_sol = opt_sol.to(device)
    #     opt_value = opt_value.to(device)
    #     predicted = net(d_batch)
    #     if model_type == "DYS":
    #         test_loss += criterion(w_batch, predicted[:,:-(num_knapsack + num_item)], opt_sol, opt_value, eval_mode=True).item()
    #     elif model_type == "Two-stage" or model_type == "SPO+" or model_type == "PertOpt-FY":
    #         test_loss += regret(w_batch, predicted, opt_sol, opt_value, eval_mode=True).item()
    #     elif model_type == "BBOpt":
    #         test_loss += criterion(w_batch, predicted, opt_sol, opt_value, eval_mode=True).item()
    #     elif model_type == "PertOpt":
    #         test_loss += criterion(w_batch, predicted, opt_sol, opt_value, eval_mode=True).item()
        
    # best_test_loss = test_loss

    ## Train!
    epoch=1
    train_time=0
    train_loss_ave = 0

    while epoch <= max_epochs and train_time <= max_time:
        start_time_epoch = time.time()
        # Iterate the training batch
        for d_batch, w_batch, opt_sol, opt_value in loader_train:
            d_batch = d_batch.to(device)
            w_batch = w_batch.to(device)
            opt_sol = opt_sol.to(device)
            opt_value = opt_value.to(device)
            net.train()
            optimizer.zero_grad()
            predicted = net(d_batch)
            if model_type == "DYS":
                loss = criterion(w_batch, predicted[:,:-(num_knapsack + num_item)], opt_sol, opt_value)
            elif model_type == "Two-stage":
                loss = criterion(w_batch, predicted)
            elif model_type == "SPO+":
                loss = criterion(predicted, w_batch, opt_sol, opt_value).mean()
            elif model_type == "BBOpt":
                x_predicted = dbb(predicted)
                loss = criterion(x_predicted, w_batch, opt_sol, opt_value)
            elif model_type == "PertOpt":
                x_predicted = ptb(predicted)
                loss = criterion(x_predicted, w_batch, opt_sol, opt_value)
            elif model_type == "PertOpt-FY":
                loss = criterion(predicted, w_batch).mean()

            loss.backward()
            optimizer.step()
            train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
        
        end_time_epoch = time.time()
        epoch_time =  end_time_epoch - start_time_epoch
        train_time += epoch_time
        epoch_time_hist.append(epoch_time)

        ## Now compute loss on validation set
        val_loss = Compute_Test_Loss(net,loader_test, model_type, metric, num_knapsack, num_item, device)

        if val_loss < best_val_loss:
            state_save_name = checkpt_path+'best.pth'
            torch.save(net.state_dict(), state_save_name)
            # If we have achieved lowest validation thus far, this will be the model selected.
            # So, we compute test loss
            best_test_loss = Compute_Test_Loss(net,loader_test, model_type, metric, num_knapsack, num_item, device)
            time_till_best_val_loss = sum(epoch_time_hist)
        
        # scheduler.step(val_loss)
        val_loss_hist.append(val_loss)
        
        print('epoch: ', epoch, 'validation loss is ', val_loss, 'epoch time: ', epoch_time)
        epoch += 1


    state_save_name = checkpt_path+'last.pth'
    torch.save(net.state_dict(), state_save_name)
    return val_loss_hist, epoch_time_hist, best_test_loss, time_till_best_val_loss