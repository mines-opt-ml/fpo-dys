from src.shortest_path import trainer, generate_shortest_path_data
from src.shortest_path.models import ShortestPathNet
import argparse
import os
import dill
import torch
import json

def main(args):
    # fetch data
    save_path = args.save_dir + 'shortest_path_training_data_' + str(args.grid_size) +'.p'
    if os.path.exists(save_path):
        file = open(save_path, 'rb')
        data = dill.load(file)
        file.close()
    else:
        print('Data not saved... generating data.')
        generate_shortest_path_data.main(args)

    # unpack data
    dataset_train = data["dataset_train"]
    dataset_test = data["dataset_test"]
    dataset_val = data["dataset_val"]
    contexts = data["contexts_numpy"]
    context_size = contexts.shape[1]

    A = data["A"]
    b = data["b"]
    edges = data["edges"]
    # context_size = 5
    # initialize model, prepare to train
    if args.model_type == "DYS":
        net = ShortestPathNet(A, b, edges, context_size)
        print(net)

    net.to(args.device)

    # Train!
    results = trainer.trainer(net, dataset_train, dataset_test, dataset_val, args.grid_size, args.learning_rate, args.model_type, args.device)

    # Dump to json
    results_path = os.path.join('./src/shortest_path/results/', 'grid_size_'+str(args.grid_size))
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    with open(os.path.join(results_path, args.model_type +'.json'), 'w') as f:
        f.write(json.dumps(results, sort_keys=True) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fpo-dys shortest path")
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./src/shortest_path/shortest_path_data/')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--model_type',type=str, default="DYS")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=10)

    args = parser.parse_args()
    main(args)
