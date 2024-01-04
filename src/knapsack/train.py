from src.knapsack import trainer, generate_knapsack_data
from src.knapsack.models import KnapSackNet, ValPredictNet
import argparse
import os
import dill
import torch
import json

def main(args):
    # fetch data
    save_path = args.save_dir + 'Knapsack_training_data_' + str(args.num_knapsack) + '_' + str(args.num_item) +'.p'
    if os.path.exists(save_path):
        file = open(save_path, 'rb')
        data = dill.load(file)
        file.close()
    else:
        print('Data not saved... generating data.')
        generate_knapsack_data.main(args)

    # unpack data
    #num_knapsack = data['num_knapsack']
    #num_item = data['num_item']
    #num_feat = data['num_feat']
    # num_data = data['num_data']
    dataset_train = data["dataset_train"]
    dataset_test = data["dataset_test"]
    dataset_val = data["dataset_val"]

    weights_numpy = data["weights_numpy"]
    capacities = data["capacities"]

    weights = torch.Tensor(weights_numpy).to(args.device)

    # initialize model, prepare to train
    if args.model_type == 'DYS':
        capacities = capacities.to(args.device)
        net = KnapSackNet(weights, capacities, args.num_knapsack, args.num_item, args.num_feat, device=args.device)
    else:
        net = ValPredictNet(args.num_knapsack, args.num_item, args.num_feat, weights, capacities, device=args.device)

    net.to(args.device)

    # Train!
    results = trainer.trainer(net, dataset_train, dataset_test, dataset_val, args.num_item, args.num_knapsack, args.max_epochs,
            args.learning_rate, args.model_type, device=args.device)

    # Dump to json
    results_path = os.path.join('./src/knapsack/results/', 'num_knapsack_'+str(args.num_knapsack))
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    with open(os.path.join(results_path, args.model_type +'.json'), 'w') as f:
        f.write(json.dumps(results, sort_keys=True) + '\n')

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="fpo-dys knapsack")
    parser.add_argument('--num_knapsack', type=int, default=2)
    parser.add_argument('--num_item', type=int, default=20)
    parser.add_argument('--num_feat', type=int, default=5)
    parser.add_argument('--num_data', type=int, default=1100)
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--model_type', type=str, default='DYS')
    parser.add_argument('--save_dir', type=str, default='./src/knapsack/knapsack_data/')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)