import json
import csv
import os
import numpy as np

## Initialize
algs = ['BBOpt', 'CVX', 'DYS', 'PertOpt']
num_algs = len(algs)
num_reps = 3
grid_sizes = [5, 10, 15, 20, 25, 30]
num_grid_sizes = len(grid_sizes)
fields = ['epoch_time_hist', 'val_loss_hist', 'time_till_best_val_loss', 'best_test_loss']

time_till_best_val_loss_matrix = np.zeros((num_reps, num_grid_sizes, num_algs))
best_test_loss_matrix = np.zeros((num_reps, num_grid_sizes, num_algs))

## Collect data from the json files
for rep in range(num_reps):
    base_path = os.path.join('./src/shortest_path/results', str(rep+1)+ '/')
    for j, grid_size in enumerate(grid_sizes):
        current_path = os.path.join(base_path, 'grid_size_'+str(grid_size))
        print(current_path)
        for k, alg in enumerate(algs):
            target_file = os.path.join(current_path, alg+'.json')
            print(target_file)
            with open(target_file) as file:
                data = json.load(file)
                time_till_best_val_loss_matrix[rep, j, k] = data['time_till_best_val_loss']
                best_test_loss_matrix[rep, j, k] = data['best_test_loss']


## Compute statistics 
mean_time_till_best_val_loss_matrix = np.mean(time_till_best_val_loss_matrix, axis=0)
max_time_till_best_val_loss_matrix = np.max(time_till_best_val_loss_matrix, axis=0)
min_time_till_best_val_loss_matrix = np.min(time_till_best_val_loss_matrix, axis=0)

mean_best_test_loss_matrix = np.mean(best_test_loss_matrix, axis=0)
max_best_test_loss_matrix = np.max(best_test_loss_matrix, axis=0)
min_best_test_loss_matrix = np.min(best_test_loss_matrix, axis=0)
print(mean_best_test_loss_matrix.shape)

## Output to csv for plotting
csv_save_path = './src/shortest_path/results/csv/'
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

# time
for k, alg in enumerate(algs):
    # mean
    target_file = os.path.join(csv_save_path, 'time_till_best_val_loss_' + alg +'_mean.csv')
    np.savetxt(target_file, np.column_stack([grid_sizes, mean_time_till_best_val_loss_matrix[:,k]]))

    # min
    target_file = os.path.join(csv_save_path, 'time_till_best_val_loss_' + alg +'_min.csv')
    np.savetxt(target_file, np.column_stack([grid_sizes, min_time_till_best_val_loss_matrix[:,k]]))

    # max
    target_file = os.path.join(csv_save_path, 'time_till_best_val_loss_' + alg +'_max.csv')
    np.savetxt(target_file, np.column_stack([grid_sizes, max_time_till_best_val_loss_matrix[:,k]]))

# best_loss
for k, alg in enumerate(algs):
    # mean
    target_file = os.path.join(csv_save_path, 'best_test_loss_' + alg +'_mean.csv')
    np.savetxt(target_file, np.column_stack([grid_sizes, mean_best_test_loss_matrix[:,k]]))

    # min
    target_file = os.path.join(csv_save_path, 'best_test_loss_' + alg +'_min.csv')
    np.savetxt(target_file, np.column_stack([grid_sizes, min_best_test_loss_matrix[:,k]]))

    # max
    target_file = os.path.join(csv_save_path, 'best_test_loss_' + alg +'_max.csv')
    np.savetxt(target_file, np.column_stack([grid_sizes, max_best_test_loss_matrix[:,k]]))






            


