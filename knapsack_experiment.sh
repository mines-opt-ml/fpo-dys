#!/bin/bash

# Define the model types
model_types="DYS CVX PertOpt BBOpt"
num_items="70 80 90 100"
reps="1 2 3"
data_dir="./src/knapsack/knapsack_data"
weights_dir="./src/knapsack/saved_weights"
results_dir="./src/knapsack/results"

# Train for each model type and grid size
for num_item in $num_items
do
    for rep in $reps
    do
        for model_type in $model_types
        do
            rep_data_dir="${data_dir}/$rep"
            rep_weights_dir="${weights_dir}/$rep"
            rep_results_dir="${results_dir}/$rep"
            python -m src.knapsack.train --model_type $model_type --num_item $num_item --num_data 1000 --data_dir $rep_data_dir --results_dir $rep_results_dir --device cuda:0
            echo "$num_item"
        done
    done
done