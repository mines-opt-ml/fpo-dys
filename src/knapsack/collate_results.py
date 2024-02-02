import json
import csv
import os
import numpy as np

## Initialize
algs = ['BBOpt', 'CVX', 'DYS', 'PertOpt']
num_algs = len(algs)
num_reps = 3
num_items = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
num_num_items = len(num_items)
fields = ['epoch_time_hist', 'val_loss_hist', 'time_till_best_val_loss', 'best_test_loss']

time_till_best_val_loss_matrix = np.zeros((num_reps, num_num_items, num_algs))
best_test_loss_matrix = np.zeros((num_reps, num_num_items, num_algs))

## Collect data from the json files
for rep in range(num_reps):
    base_path = os.path.join('./src/knapsack/results/deg_2', str(rep+1)+ '/')
    for j, num_item in enumerate(num_items):
        current_path = os.path.join(base_path, 'num_knapsack_'+str(num_item))
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
csv_save_path = './src/knapsack/results/csv/'
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

# time
for k, alg in enumerate(algs):
    # mean
    target_file = os.path.join(csv_save_path, 'time_till_best_val_loss_' + alg +'_mean.csv')
    np.savetxt(target_file, np.column_stack([num_items, mean_time_till_best_val_loss_matrix[:,k]]), delimiter=",")

    # min
    target_file = os.path.join(csv_save_path, 'time_till_best_val_loss_' + alg +'_min.csv')
    np.savetxt(target_file, np.column_stack([num_items, min_time_till_best_val_loss_matrix[:,k]]), delimiter=",")

    # max
    target_file = os.path.join(csv_save_path, 'time_till_best_val_loss_' + alg +'_max.csv')
    np.savetxt(target_file, np.column_stack([num_items, max_time_till_best_val_loss_matrix[:,k]]), delimiter=",")

# best_loss
for k, alg in enumerate(algs):
    # mean
    target_file = os.path.join(csv_save_path, 'best_test_loss_' + alg +'_mean.csv')
    np.savetxt(target_file, np.column_stack([num_items, mean_best_test_loss_matrix[:,k]]), delimiter=",")

    # min
    target_file = os.path.join(csv_save_path, 'best_test_loss_' + alg +'_min.csv')
    np.savetxt(target_file, np.column_stack([num_items, min_best_test_loss_matrix[:,k]]), delimiter=",")

    # max
    target_file = os.path.join(csv_save_path, 'best_test_loss_' + alg +'_max.csv')
    np.savetxt(target_file, np.column_stack([num_items, max_best_test_loss_matrix[:,k]]), delimiter=",")






            


