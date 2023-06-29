import numpy as np
import numpy.random as rand
import torch
import time as time
from ModelsKnapSack import KnapSackNet, ValPredictNet
from Trainer import trainer

# Set the random seed
seed = rand.randint(0, 512)

def _initializer(knapsack_dict, knapsack_data_dict, model_type, device='mps'):
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
    if model_type == 'DYS':
        capacities = capacities.to(device)
        net  = KnapSackNet(weights, capacities, num_knapsack, num_item, num_feat, device=device)
    else:
        net = ValPredictNet(num_knapsack, num_item, num_feat, weights, capacities, device=device)
    net.to(device)
    return net, num_item, num_knapsack

def initialize_and_train(knapsack_dict, knapsack_data_dict, model_type, max_epochs, learning_rate=1e-3, device='mps'):
    net, num_item, num_knapsack = _initializer(knapsack_dict, knapsack_data_dict, model_type, device='mps')
    dataset_train = knapsack_dict["dataset_train"]
    dataset_test = knapsack_dict["dataset_test"]
    dataset_val = knapsack_dict["dataset_val"]
    #if model_type == "Two-stage":
    #    max_epochs = 2*max_epochs
    
    print('\n Currently training ' + model_type + '\n')
    test_loss_hist, epoch_time_hist, best_test_loss, time_till_best_test_loss = trainer(net, dataset_train, dataset_test, dataset_val, num_item, num_knapsack, max_epochs, learning_rate, model_type = model_type, device=device)
    return np.sum(epoch_time_hist), best_test_loss, time_till_best_test_loss
