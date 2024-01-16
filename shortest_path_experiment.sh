#!/bin/bash

# Define the model types
model_types="DYS PertOpt BBOpt"
grid_sizes="5 10 15 20 25 30"
reps="1 2 3"
data_dir="./src/shortest_path/shortest_path_data"
weights_dir="./src/shortest_path/saved_weights"
results_dir="./src/shortest_path/results"

# Train for each model type and grid size
for grid_size in $grid_sizes
do
    for rep in $reps
    do
        for model_type in $model_types
        do
            rep_data_dir="${data_dir}/$rep"
            rep_weights_dir="${weights_dir}/$rep"
            rep_results_dir="${results_dir}/$rep"
            python -m src.shortest_path.train --model_type $model_type --grid_size $grid_size --num_data 1000 --data_dir $rep_data_dir --results_dir $rep_results_dir --device cuda:0
            echo "$grid_size"
        done
    done
done