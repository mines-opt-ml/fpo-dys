import numpy as np
import os

def main(args):
    '''
When called, this script prepares the warcraft data files into convenient data loaders.
It is written in this form, although there are no important args, for consistency with 
the other experiments.
'''
    grid_size = 12
    tmaps_train = np.load(os.join(args.source_dir, '{}x{}/train_maps.npy'.format(grid_size, grid_size)))
    tmaps_test = np.load(os.join(args.source_dir, '{}x{}/test_maps.npy'.format(grid_size, grid_size)))
    tmaps_val = np.load(os.join(args.source_dir, '{}x{}/val_maps.npy'.format(grid_size, grid_size)))

    costs_train = np.load(os.join(args.source_dir, '{}x{}/train_vertex_weights.npy'.format(grid_size, grid_size)))
    costs_test = np.load(os.join(args.source_dir, '{}x{}/test_vertex_weights.npy'.format(grid_size, grid_size)))
    costs_val = np.load(os.join(args.source_dir, '{}x{}/val_vertex_weights.npy'.format(grid_size, grid_size)))

    paths_train = np.load(os.join(args.source_dir, '{}x{}/train_shortest_paths.npy'.format(grid_size, grid_size)))
    paths_test = np.load(os.join(args.source_dir, '{}x{}/test_shortest_paths.npy'.format(grid_size, grid_size)))
    paths_val = np.load(os.join(args.source_dir, '{}x{}/val_shortest_paths.npy'.format(grid_size, grid_size)))
    
