from sklearn.model_selection import train_test_split
from pyepo.model.grb import shortestPathModel
import pyepo
import torch
import dill
import argparse
import os

# generate actual data
def main(args):
    
    # unpack args
    grid_size = args.grid_size
    grid = (grid_size, grid_size)
    num_data = args.num_data
    num_feat = args.num_feat

    print('Generating training data for shortest path with '+str(grid_size)
          + '-by-' + str(grid_size) + ' grid.')
    
    contexts_numpy, costs_numpy = pyepo.data.shortestpath.genData(num_data + 400, num_feat, grid, deg=4, noise_width= 0.5)
    
    # split train test data
    d_train, d_test_val, w_train, w_test_val = train_test_split(contexts_numpy, costs_numpy, test_size=400)
    d_test, d_val, w_test, w_val = train_test_split(d_test_val, w_test_val, test_size=200)
    
    # Define PyEPO model
    optmodel = shortestPathModel(grid)

    # get optDataset
    dataset_train = pyepo.data.dataset.optDataset(optmodel, d_train, w_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, d_test, w_test)
    dataset_val = pyepo.data.dataset.optDataset(optmodel, d_val, w_val)

    # get edges/arcs
    edges = optmodel.arcs

    # contruct edge-node incidence matrix A and flow vector b
    A = torch.zeros((grid_size**2, len(edges)))
    for j,e in enumerate(edges):
        ind0 = e[0]
        ind1 = e[1]
        A[ind0, j] = -1.
        A[ind1, j] = +1.

    b = torch.zeros(grid_size**2)
    b[0] = -1.
    b[-1] = 1.

    # Remove the gurobi model before saving
    dataset_train.model = None
    dataset_test.model = None
    dataset_val.model = None

    # Package into a dictionary
    state = {'edges': edges,
             'A': A,
             'b': b,
             'contexts_numpy': contexts_numpy,
             'costs_numpy'   : costs_numpy,
             'dataset_train' : dataset_train,
             'dataset_test'  : dataset_test,
             'dataset_val'   : dataset_val, 
            }

    # Save and finish up
    print('Finished building dataset')
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    state_path = os.path.join(args.data_dir, 'shortest_path_training_data_' + str(grid_size) + '.p')
    dill.dump(state, open( state_path, "wb" ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate shortest path data')
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--num_feat', type=int, default=5)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./src/shortest_path/shortest_path_data/')
    args = parser.parse_args()
    main(args)


    

    
    
