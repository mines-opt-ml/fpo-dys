from sklearn.model_selection import train_test_split
import pyepo
import torch
import dill
import argparse
import os

# generate actual data
def main(args):
    
    # unpack args
    num_knapsack = args.num_knapsack
    num_data = args.num_data
    num_feat = args.num_feat
    num_item = args.num_item

    capacities = 20*torch.ones(num_knapsack)
    print('Generating training data for knapsack problem with '+str(num_knapsack)
          + ' knapsacks and ' + str(num_item) + ' items')
    
    weights_numpy, contexts_numpy, costs_numpy = pyepo.data.knapsack.genData(num_data, num_feat, num_item,
                                                dim=num_knapsack, deg=args.data_deg, noise_width=args.data_noise_width)
    
    # split train test data
    d_train, d_test_val, w_train, w_test_val = train_test_split(contexts_numpy, costs_numpy, test_size=200)
    d_test, d_val, w_test, w_val = train_test_split(d_test_val, w_test_val, test_size=100)
    
    # Define PyEPO model
    caps = capacities.cpu() #[20] * 2 # capacity
    optmodel = pyepo.model.grb.knapsackModel(weights_numpy, caps)

    # get optDataset
    dataset_train = pyepo.data.dataset.optDataset(optmodel, d_train, w_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, d_test, w_test)
    dataset_val = pyepo.data.dataset.optDataset(optmodel, d_val, w_val)

    # Remove the gurobi model befor esaving
    dataset_train.model = None
    dataset_test.model = None
    dataset_val.model = None

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
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    state_path = args.data_dir + 'Knapsack_training_data_' + str(num_knapsack) + '_' + str(num_item) +'.p'
    dill.dump( state, open( state_path, "wb" ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate knapsack data')
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--num_feat', type=int, default=5)
    parser.add_argument('--num_item', type=int, default=20)
    parser.add_argument('-num_knapsack', type=int, default=2)
    parser.add_argument('--data_deg', type=int, default=4)
    parser.add_argument('--data_noise_width', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='./src/knapsack/knapsack_data/')
    args = parser.parse_args()
    main(args)


    

    
    
