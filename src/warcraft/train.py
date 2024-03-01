from src.warcraft import trainer
from src.warcraft.models import WarcraftShortestPathNet
import argparse
import os
import dill
import torch
import json
import numpy as np

def main(args):
    # fetch data
    print(args.data_dir)
    data_path = os.path.join(args.data_dir,'warcraft_data_12.p')
    if os.path.exists(data_path):
        file = open(data_path, 'rb')
        data = dill.load(file)
        file.close()
    else:
        print('\n Data not found. Download data as described in ReadMe and process with generate_warcraft_data \n Script terminating ... \n')
        return

    # unpack data
    dataset_train = data["dataset_train"]
    dataset_test = data["dataset_test"]
    dataset_val = data["dataset_val"]
    edges = data["edges"]

    A = data["A"].to(args.device)
    b = data["b"].to(args.device)
   
    # initialize model, prepare to train
    if args.model_type == "DYS":
        net = WarcraftShortestPathNet(args.grid_size, A, b, args.device)
    else:
        print('\n Other models not implemented yet, sorry. \n')
        return

    net.to(args.device)

    # Train!
    print('\n---- Model type= ' + args.model_type + ' Grid size = ' + str(args.grid_size) + '---\n')
    results = trainer.trainer(net, dataset_train, dataset_test, dataset_val, edges, args.grid_size, args.max_time, args.max_epochs, args.learning_rate, args.model_type, args.weights_dir, args.device)

    # Dump to json
    results_path = os.path.join(args.results_dir, 'grid_size_'+str(args.grid_size))
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    with open(os.path.join(results_path, args.model_type +'.json'), 'w') as f:
        f.write(json.dumps(results, sort_keys=True) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fpo-dys warcraft")
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--data_dir', type=str, default='./src/warcraft/warcraft_data/')
    parser.add_argument('--weights_dir', type=str, default='./src/warcraft/saved_weights/')
    parser.add_argument('--results_dir', type=str, default='./src/warcraft/results/')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--model_type',type=str, default="DYS")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_time', type=int, default=1800)

    args = parser.parse_args()
    main(args)
