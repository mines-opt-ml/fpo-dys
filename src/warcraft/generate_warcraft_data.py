import numpy as np
import torch
import os
import dill
import argparse
from torch.utils.data import Dataset, DataLoader

class mapDataset(Dataset):
    '''
    From PyEPO demo notebook.
    '''
    def __init__(self, tmaps, costs, paths):
        self.tmaps = tmaps
        self.costs = costs
        self.paths = paths
        self.objs = (costs * paths).sum(axis=(1,2)).reshape(-1,1)
        
    def __len__(self):
        return len(self.costs)
    
    def __getitem__(self, ind):
        return (
            torch.FloatTensor(self.tmaps[ind].transpose(2, 0, 1)/255).detach(), # image
            torch.FloatTensor(self.costs[ind]).reshape(-1),
            torch.FloatTensor(self.paths[ind]).reshape(-1),
            torch.FloatTensor(self.objs[ind]),
        )

def main(args):
    '''
When called, this script prepares the warcraft data files into convenient data loaders.
It is written in this form, although there are no important args, for consistency with 
the other experiments.
'''
    grid_size = 12
    tmaps_train = np.load(os.path.join(args.source_dir, '{}x{}/train_maps.npy'.format(grid_size, grid_size)))
    tmaps_test = np.load(os.path.join(args.source_dir, '{}x{}/test_maps.npy'.format(grid_size, grid_size)))
    tmaps_val = np.load(os.path.join(args.source_dir, '{}x{}/val_maps.npy'.format(grid_size, grid_size)))

    costs_train = np.load(os.path.join(args.source_dir, '{}x{}/train_vertex_weights.npy'.format(grid_size, grid_size)))
    costs_test = np.load(os.path.join(args.source_dir, '{}x{}/test_vertex_weights.npy'.format(grid_size, grid_size)))
    costs_val = np.load(os.path.join(args.source_dir, '{}x{}/val_vertex_weights.npy'.format(grid_size, grid_size)))

    paths_train = np.load(os.path.join(args.source_dir, '{}x{}/train_shortest_paths.npy'.format(grid_size, grid_size)))
    paths_test = np.load(os.path.join(args.source_dir, '{}x{}/test_shortest_paths.npy'.format(grid_size, grid_size)))
    paths_val = np.load(os.path.join(args.source_dir, '{}x{}/val_shortest_paths.npy'.format(grid_size, grid_size)))

    # datasets
    dataset_train = mapDataset(tmaps_train, costs_train, paths_train)
    dataset_val = mapDataset(tmaps_val, costs_val, paths_val)
    dataset_test = mapDataset(tmaps_test, costs_test, paths_test)


    # Package into a dictionary
    state = {'grid_size'     : grid_size,
             'dataset_train' : dataset_train,
             'dataset_test'  : dataset_test,
             'dataset_val'   : dataset_val, 
            }

    # Save and finish up
    print('Finished building dataset')
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    state_path = os.path.join(args.data_dir, 'warcraft_data_' + str(grid_size) + '.p')
    dill.dump(state, open(state_path, "wb" ))

    # # create dataloaders
    # batch_size = 64
    # loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    # loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate warcraft data')
    parser.add_argument('--source_dir', type=str, default='./src/warcraft/raw_data/')
    parser.add_argument('--data_dir', type=str, default='./src/warcraft/warcraft_data/')
    args = parser.parse_args()
    main(args)

    

