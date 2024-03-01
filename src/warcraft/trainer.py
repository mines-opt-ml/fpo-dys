import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import time as time
import torch.nn as nn
import pyepo
import os
from src.shortest_path.utils import edge_to_node
from src.shortest_path.shortest_path_utils import convert_to_grid_torch, evaluate 
from src.utils.accuracy import accuracy
from src.utils.evaluate import evaluate
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

def trainer(net, train_dataset, test_dataset, val_dataset, edges, grid_size, max_time, max_epochs, learning_rate, model_type, weights_dir, device='mps'):

    ## Training setup
    batch_size = 70
    loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True)
    loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False)
    loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Initialize loss and evaluation metric
    if model_type == "DYS" or model_type == "CVX":
        criterion = nn.MSELoss()
    elif model_type == "BBOpt" or model_type == "PertOpt":
        criterion = nn.L1Loss()

    metric = pyepo.metric.regret

    if model_type == "BBOpt":
        dbb = pyepo.func.blackboxOpt(net.shortest_path_solver, lambd=5, processes=1)
    elif model_type == "PertOpt":
        ptb = pyepo.func.perturbedOpt(net.shortest_path_solver, n_samples=3, sigma=1.0, processes=2)
    elif model_type == "DYS" or model_type == "CVX":
        pass
    else:
        raise TypeError("Please choose a supported model!")

    ## Initialize arrays that will be returned and checkpoint directory
    val_loss_hist= []
    test_regret_hist = []
    test_acc_hist = []
    val_acc_hist = []
    epoch_time_hist = [0]
    train_mse_loss_hist = []
    checkpt_path = weights_dir + model_type + '/'
    if not os.path.exists(checkpt_path):
        os.makedirs(checkpt_path)

    evaluate(net, net.shortest_path_solver, loader_test)
    net.eval()
    net.to('cpu')
    curr_val_acc = accuracy(net, net.shortest_path_solver, loader_val)
    val_acc_hist.append(curr_val_acc)
    best_val_loss = metric(net, net.shortest_path_solver, loader_val)

    print('Initial validation regret is ', best_val_loss)
    val_loss_hist.append(best_val_loss)
    time_till_best_val_loss = 0

     ## Compute initial test loss
    best_test_loss = metric(net,net.shortest_path_solver, loader_test)
    best_test_acc = accuracy(net, net.shortest_path_solver, loader_test)
    print('Initial test regret is ', best_test_loss)
    test_regret_hist.append(best_test_loss)
    test_acc_hist.append(best_test_acc)
   
    ## Train!
    train_time=0
    train_loss_ave = 0
    tbar = tqdm.tqdm(range(max_epochs))
    
    #while epoch <= max_epochs and train_time <= max_time:
    for epoch in tbar:
        start_time_epoch = time.time()
        net.to(device)
        batch_counter = 0
        # Iterate the training batch
        for d_batch, w_batch, opt_sol, opt_value in loader_train:
            d_batch = d_batch.to(device)
            w_batch = w_batch.to(device)
            opt_sol = opt_sol.to(device)
            opt_value = opt_value.to(device)
            net.train()
            predicted = net(d_batch)
            #print(edge_to_node(predicted[1,:], edges, grid_size, device))
            #print(edge_to_node(opt_sol[1,:], edges, grid_size, device))
            if model_type == "DYS" or model_type == "CVX":
                grid_predicted, predicted_reshaped = convert_to_grid_torch(predicted, grid_size, edges, net.shortest_path_solver.nodes_map, net.device)
                # if batch_counter % 50 == 0:
                #     rand_number = np.random.randint(predicted.shape[0])
                #     fig, axes = plt.subplots(1,2)
                #     cax1 = axes[0].matshow(opt_sol[rand_number,:].reshape(12,12).cpu().detach().numpy())
                #     axes[0].set_title('True')

                #     cax2 = axes[1].matshow(grid_predicted[rand_number,:,:].cpu().detach().numpy())
                #     axes[1].set_title('Predicted')

                #     plt.savefig(os.path.join('./src/warcraft/imgs/','Comparing_output_epoch='+str(epoch)+'_batch_id='+str(rand_number)+".png"))
                #     plt.close()

                # print('\n ---------------- \n True \n ')
                # print(opt_sol[6,:].reshape(12,12).cpu().detach().numpy())
                # print('\n Prediction \n')
                # print(np.round(grid_predicted[6,:,:].cpu().detach().numpy()))
                # print('----------------------------')
                # print(np.vstack((opt_sol[4,:].cpu().detach().numpy(), np.round(predicted_reshaped[4,:].cpu().detach().numpy()))))

                # print(grid_predicted[1,:,:])
                # print('\n Opt sol is \n')
                # print(opt_sol[1,:])
                loss = criterion(opt_sol, predicted_reshaped)
            elif model_type == "BBOpt":
                x_predicted = dbb(predicted)
                loss = criterion(opt_sol, x_predicted)
            elif model_type == "PertOpt":
                x_predicted = ptb(predicted)
                loss = criterion(opt_sol, x_predicted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mse_loss_hist.append(loss.item())
            tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
            #train_loss_ave = 0.95*train_loss_ave + 0.05*loss.item()
            #print(f"\n Training loss is {loss.item()}")
            batch_counter += 1
        
        end_time_epoch = time.time()
        epoch_time =  end_time_epoch - start_time_epoch
        train_time += epoch_time
        epoch_time_hist.append(epoch_time)

        ## Now compute loss on validation set
        net.eval()
        net.to('cpu')
        val_loss = metric(net, net.shortest_path_solver, loader_val)
        val_acc = accuracy(net, net.shortest_path_solver, loader_val)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)
        ## Do the same on test set for consistency with PyEPO
        test_loss = metric(net, net.shortest_path_solver, loader_test)
        test_acc = accuracy(net, net.shortest_path_solver, loader_test)
        test_acc_hist.append(test_acc)
        test_regret_hist.append(test_loss)

        print('\n Current validation accuracy is ' + str(val_acc))
        if (epoch == int(max_epochs*0.6)) or (epoch == int(max_epochs*0.8)):
            for g in optimizer.param_groups:
                g['lr'] /= 10

        if val_loss < best_val_loss:
            state_save_name = checkpt_path+'best.pth'
            torch.save(net.state_dict(), state_save_name)
            # If we have achieved lowest validation thus far, this will be the model selected.
            # So, we compute test loss
            best_test_loss = metric(net,net.shortest_path_solver, loader_test)
            best_test_acc = accuracy(net, net.shortest_path_solver, loader_test)
            best_val_loss = val_loss
            print(f'Best test regret is {best_test_loss} achieved at epoch {epoch}')
            time_till_best_val_loss = sum(epoch_time_hist)
        
        print('epoch: ', epoch, 'validation regret is ', val_loss, 'epoch time: ', epoch_time)

    state_save_name = checkpt_path+'last.pth'
    torch.save(net.state_dict(), state_save_name)
    if time_till_best_val_loss < 1e-6:
        time_till_best_val_loss = sum(epoch_time_hist)

    # Collect results
    results = {"val_loss_hist": val_loss_hist,
               "train_mse_loss_hist": train_mse_loss_hist,
               "val_acc_hist": val_acc_hist,
               "test_regret_hist": test_regret_hist,
               "test_acc_hist": test_acc_hist,
               "epoch_time_hist": epoch_time_hist,
               "best_test_loss": best_test_loss,
               "best_test_acc": best_test_acc,
               "time_till_best_val_loss":time_till_best_val_loss
                }
        
    return results