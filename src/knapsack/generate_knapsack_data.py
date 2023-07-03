from sklearn.model_selection import train_test_split
import pyepo
import torch
import pickle as pkl

# generate actual data
def Gen_Knapsack_data(num_data = 10000, num_feat = 5, num_item = 20, num_knapsack = 2):
    capacities = 20*torch.ones(num_knapsack)
    print('Generating training data for knapsack problem with '+str(num_knapsack)
          + ' knapsacks and ' + str(num_item) + ' items')
    # split train test data
    weights_numpy, contexts_numpy, costs_numpy = pyepo.data.knapsack.genData(num_data, num_feat, num_item,
                                                dim=num_knapsack, deg=4, noise_width= 0.5)#0.5)#, seed=135)

    d_train, d_test_val, w_train, w_test_val = train_test_split(contexts_numpy, costs_numpy, test_size=200)#, random_state=42)
    d_test, d_val, w_test, w_val = train_test_split(d_test_val, w_test_val, test_size=100)
    
    # Define PyEPO model
    caps = capacities.cpu() #[20] * 2 # capacity
    optmodel = pyepo.model.grb.knapsackModel(weights_numpy, caps)

    # get optDataset
    dataset_train = pyepo.data.dataset.optDataset(optmodel, d_train, w_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, d_test, w_test)
    dataset_val = pyepo.data.dataset.optDataset(optmodel, d_val, w_val)
    # dataset_validation = pyepo.data.dataset.optDataset(optmodel, d_test, w_test)

    # Package into a dictionary
    state = { 'weights_numpy' : weights_numpy,
              'contexts_numpy': contexts_numpy,
              'costs_numpy'   : costs_numpy,
              'dataset_train' : dataset_train,
              'dataset_test'  : dataset_test,
              'dataset_val'   : dataset_val,
              'capacities'    : capacities, 
            }

    # Save and finish up
    print('Finished building dataset')
    return state
    # state_path = save_dir + 'Knapsack_training_data' + str(num_knapsack) + '_' + str(num_item) +'.pth'
    # pkl.dump( state, open( state_path, "wb" ) )
    
    
