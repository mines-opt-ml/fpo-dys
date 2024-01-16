from src.shortest_path import trainer, generate_shortest_path_data
from src.shortest_path.models import ShortestPathNet, Generic_ShortestPathNet, Cvx_ShortestPathNet
import argparse
import os
import dill
import torch
import json

def main(args):
    # fetch data
    print(args.data_dir)
    data_path = os.path.join(args.data_dir, 'shortest_path_training_data_' + str(args.grid_size) +'.p')
    if os.path.exists(data_path):
        file = open(data_path, 'rb')
        data = dill.load(file)
        file.close()
    else:
        print('Data not saved... generating data.')
        generate_shortest_path_data.main(args)
        file = open(data_path, 'rb')
        data = dill.load(file)
        file.close()

    # unpack data
    dataset_train = data["dataset_train"]
    dataset_test = data["dataset_test"]
    dataset_val = data["dataset_val"]
    contexts = data["contexts_numpy"]
    context_size = contexts.shape[1]
    edges = data["edges"]

    A = data["A"].to(args.device)
    b = data["b"].to(args.device)
    
    # initialize model, prepare to train
    if args.model_type == "DYS":
        net = ShortestPathNet(args.grid_size, A, b, edges, context_size, args.device)
    elif args.model_type == "BBOpt" or args.model_type == "PertOpt":
        net = Generic_ShortestPathNet(A, context_size, args.grid_size, args.device)
        print(net)
    elif args.model_type == "CVX": 
        net = Cvx_ShortestPathNet(args.grid_size, A, b, context_size, args.device)

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
    parser = argparse.ArgumentParser(description="fpo-dys shortest path")
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./src/shortest_path/shortest_path_data/')
    parser.add_argument('--data_deg', type=int, default=4)
    parser.add_argument('--data_noise_width', type=float, default=0.5)
    parser.add_argument('--weights_dir', type=str, default='./src/shortest_path/saved_weights/')
    parser.add_argument('--results_dir', type=str, default='./src/shortest_path/results/')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--model_type',type=str, default="DYS")
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--num_feat', type=int, default=5)
    parser.add_argument('--max_time', type=int, default=1800)

    args = parser.parse_args()
    main(args)
