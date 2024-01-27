'''
Trainer for knapsack problem. This implements the "x" version of the problem where the training 
data is pairs (d, x(d)).

Daniel McKenzie
June 2023

'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import time as time
import torch.nn as nn
from src.knapsack.knapsack_utils import RegretLoss, Compute_Test_Loss
import pyepo
import os

def trainer(net, train_dataset, test_dataset, val_dataset, num_item, num_knapsack, max_time, max_epochs, learning_rate, model_type, weights_dir, device='mps'):
   

    ## Training setup
    batch_size = 256
    loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True)
    loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False)
    loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 5e-4)

    # Initialize loss function and evaluation metric
    metric = pyepo.metric.regret
   
    if model_type == "DYS" or model_type == "CVX":
        criterion = nn.MSELoss()
    elif model_type == "BBOpt" or model_type == "PertOpt":
        criterion = nn.L1Loss()
    else:
        raise TypeError("Please choose a supported model!")
       
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if model_type == "BBOpt":
        dbb = pyepo.func.blackboxOpt(net.knapsack_solver, lambd=5, processes=1)
    elif model_type == "PertOpt":
        ptb = pyepo.func.perturbedOpt(net.knapsack_solver, n_samples=3, sigma=1.0, processes=2)
    elif model_type == "DYS" or model_type == "CVX":
        pass
    else:
        raise TypeError("Please choose a supported model!")

    ## Initialize arrays that will be returned and checkpoint directory
    val_loss_hist   = []
    epoch_time_hist = []

    checkpt_path = weights_dir + model_type + '/'
    if not os.path.exists(checkpt_path):
        os.makedirs(checkpt_path)

    net.eval()
    net.to('cpu')
    best_val_loss = metric(net, net.knapsack_solver, loader_val)

    print('Initial validation loss is ', best_val_loss)
    val_loss_hist.append(best_val_loss)
    time_till_best_val_loss = 0

     ## Compute initial test loss
    best_test_loss = metric(net, net.knapsack_solver, loader_test)
   
    ## Train!
    epoch=1
    train_time=0
    train_loss_ave = 0

    while epoch <= max_epochs and train_time <= max_time:
        start_time_epoch = time.time()
        net.to(device)
        # Iterate the training batch
        for d_batch, w_batch, opt_sol, opt_value in loader_train:
            d_batch = d_batch.to(device)
            w_batch = w_batch.to(device)
            opt_sol = opt_sol.to(device)
            opt_value = opt_value.to(device)
            net.train()
            optimizer.zero_grad()
            predicted = net(d_batch)
            if model_type == "DYS" or model_type == "CVX":
                loss = criterion(opt_sol, predicted[:,:-(num_knapsack + num_item)])
            elif model_type == "BBOpt":
                x_predicted = dbb(predicted)
                loss = criterion(opt_sol, x_predicted)
            elif model_type == "PertOpt":
                x_predicted = ptb(predicted)
                loss = criterion(opt_sol, x_predicted)

            loss.backward()
            optimizer.step()
            train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
        
        end_time_epoch = time.time()
        epoch_time =  end_time_epoch - start_time_epoch
        train_time += epoch_time
        epoch_time_hist.append(epoch_time)

        ## Now compute loss on validation set
        net.eval()
        net.to('cpu')
        val_loss = metric(net, net.knapsack_solver, loader_val)
        print('\n Current validation loss is ' + str(val_loss))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            state_save_name = checkpt_path+'best.pth'
            torch.save(net.state_dict(), state_save_name)
            # If we have achieved lowest validation thus far, this will be the model selected.
            # So, we compute test loss
            best_test_loss = metric(net, net.knapsack_solver, loader_test)
            best_val_loss = val_loss
            print('Best validation loss achieved at epoch ' + str(epoch))
            time_till_best_val_loss = sum(epoch_time_hist)
        
        val_loss_hist.append(val_loss)
        
        print('epoch: ', epoch, 'validation loss is ', val_loss, 'epoch time: ', epoch_time)
        epoch += 1


    state_save_name = checkpt_path+'last.pth'
    torch.save(net.state_dict(), state_save_name)
    if time_till_best_val_loss < 1e-6:
        time_till_best_val_loss = sum(epoch_time_hist)

    # Collect results
    results = {"val_loss_hist": val_loss_hist,
                "epoch_time_hist": epoch_time_hist,
                "best_test_loss": best_test_loss,
                "time_till_best_val_loss":time_till_best_val_loss
                }
        
    return results
