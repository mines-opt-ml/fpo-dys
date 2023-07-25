import numpy as np
import numpy.random as rand
import torch
import time as time
from ModelsKnapSack import KnapSackNet, ValPredictNet
from Trainer_w import trainer_w
from Trainer_x import trainer_x

# Set the random seed
seed = rand.randint(0, 512)

def _initializer(knapsack_dict, knapsack_data_dict, model_type, device='cuda:0'):
    # unpack knapsack problem data
    num_knapsack = knapsack_data_dict['num_knapsack']
    num_item = knapsack_data_dict['num_item']
    num_feat = knapsack_data_dict['num_feat']
    num_data = knapsack_data_dict['num_data']

    weights_numpy = knapsack_dict["weights_numpy"]
    capacities = knapsack_dict["capacities"]

    weights = torch.Tensor(weights_numpy).to(device)

    # fix seed
    torch.manual_seed(seed)

    # initialize model
    if model_type == 'DYS' or model_type == 'DYS-Regret':
        capacities = capacities.to(device)
        net  = KnapSackNet(weights, capacities, num_knapsack, num_item, num_feat, device=device)
    else:
        net = ValPredictNet(num_knapsack, num_item, num_feat, weights, capacities, device=device)
    net.to(device)
    return net, num_item, num_knapsack

def initialize_and_train(knapsack_dict, knapsack_data_dict, model_type, data_type, max_epochs, learning_rate=1e-3, device='cuda:0'):
    net, num_item, num_knapsack = _initializer(knapsack_dict, knapsack_data_dict, model_type, device='cuda:0')
    dataset_train = knapsack_dict["dataset_train"]
    dataset_test = knapsack_dict["dataset_test"]
    dataset_val = knapsack_dict["dataset_val"]
    #if model_type == "Two-stage":
    #    max_epochs = 2*max_epochs
    
    print('\n Currently training ' + model_type + '\n')
    if data_type == "x":
        test_loss_hist, epoch_time_hist, best_test_loss, time_till_best_test_loss = trainer_x(net, dataset_train, dataset_test, dataset_val, num_item, num_knapsack, max_epochs, learning_rate, model_type = model_type, device=device)
    elif data_type == "w":
        test_loss_hist, epoch_time_hist, best_test_loss, time_till_best_test_loss = trainer_w(net, dataset_train, dataset_test, dataset_val, num_item, num_knapsack, max_epochs, learning_rate, model_type = model_type, device=device)
    else:
        TypeError('Please choose a supported data type!')
    return np.sum(epoch_time_hist), best_test_loss, time_till_best_test_loss
