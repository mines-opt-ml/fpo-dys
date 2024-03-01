import numpy as np
from src.warcraft.utils import evaluate, plotLearningCurve
import json
from src.warcraft.models import WarcraftShortestPathNet
from torch.utils.data import DataLoader
import torch
import os
import dill
import time

device = 'cuda:0'

## Load the training statistics
target_file = 'src/warcraft/results/grid_size_12/DYS.json'
print(target_file)
with open(target_file) as file:
    data = json.load(file)

## Fetch training data loader
data_path = 'src/warcraft/warcraft_data/warcraft_data_12.p'
file = open(data_path, 'rb')
net_data = dill.load(file)
file.close()

## unpack data
dataset_train = net_data["dataset_train"]
dataset_test = net_data["dataset_test"]
dataset_val = net_data["dataset_val"]
edges = net_data["edges"]

A = net_data["A"].to(device)
b = net_data["b"].to(device)

 ## Training setup
batch_size = 70
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                shuffle=True)
loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                shuffle=False)
loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size,
                                shuffle=False)

## Load the model and weights
nnet = WarcraftShortestPathNet(12, A, b, 'cuda:0')
weights_path = 'src/warcraft/saved_weights/DYS/best.pth'
nnet.load_state_dict(torch.load(weights_path))

train_mse_loss_hist = data["train_mse_loss_hist"]
test_regret_hist = data['test_regret_hist']
epoch_time_hist = data['epoch_time_hist']
#epoch_time_hist.insert(0, 0)
epoch_time = np.cumsum(epoch_time_hist)

plotLearningCurve(train_mse_loss_hist, test_regret_hist, epoch_time, 50)
evaluate(nnet, nnet.shortest_path_solver, loader_test)