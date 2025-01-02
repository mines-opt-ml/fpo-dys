import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import dill

def main(args):
    '''' Utility function for viewing the predicted cost vector explicitly as a grid.

        Args:
            args (argparse): Command line arguments.

        Returns:
            Writes results to json file with location specified
    '''
    # fetch data
    print(args.data_dir)
    data_path = os.path.join(args.data_dir, 'shortest_path_training_data_' + str(args.grid_size) +'.p')
    if os.path.exists(data_path):
        file = open(data_path, 'rb')
        data = dill.load(file)
        file.close()
    else:
        print('Sorry. Could not find data for this grid size')
        return

    # unpack data
    cost_vecs = data["costs_numpy"]
    edges = data["edges"]

    # Convert costs to node format and plot
    idxs = [1, 27]
    for num, idx in enumerate(idxs):
        cost_vec = cost_vecs[idx]
        cost_mat = np.zeros((12,12))
        cost_mat[0,0] = 1.
        for i, e in enumerate(edges):
            _, node_1 = e
            reindex_node_1 = (node_1//args.grid_size, node_1 % args.grid_size)
            cost_mat[reindex_node_1] += cost_vec[i]
        # plot and save
        plt.axis("off")
        plt.imshow(cost_mat)
        plt.savefig(os.path.join('./src/shortest_path/imgs/', 'pyepo_shortest_path_cost_matrix_'+str(num)+'.pdf'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shortest path cost plot")
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--data_dir', type=str, default='./src/shortest_path/shortest_path_data/')

    args = parser.parse_args()
    main(args)



