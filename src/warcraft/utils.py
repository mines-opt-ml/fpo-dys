import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import os
import time

def plotLearningCurve(loss_log, regret_log, epoch_time, epochs):
    # draw loss during training
    plt.figure(figsize=(8, 4))
    plt.plot(loss_log, color="c")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(-100, len(loss_log)+100)
    plt.xlabel("Iters", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Learning Curve on Training Set", fontsize=12)
    plt.savefig(os.path.join('./src/warcraft/imgs/','learning_curve.png'))
    # draw normalized regret on test
    plt.figure(figsize=(8, 4))
    plt.plot([i for i in range(len(regret_log))], regret_log, color="royalblue")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(-epochs/50, epochs+epochs/50)
    plt.ylim(0, max(regret_log[1:])*1.1)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Normalized Regret", fontsize=12)
    plt.title("Learning Curve on Test Set", fontsize=12)
    plt.savefig(os.path.join('./src/warcraft/imgs/','normalized_regret_curve.png'))
    # draw regret vs epoch time
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_time, regret_log, color="orange")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(0, max(epoch_time[1:])*1.1)
    plt.ylim(0, max(regret_log[1:])*1.1)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Regret", fontsize=12)
    plt.title("Regret on Test Set vs Time")
    plt.savefig(os.path.join('./src/warcraft/imgs/','normalized_regret_vs_time_curve.png'))


def evaluate(nnet, optmodel, dataloader):
    # init data
    data = {"Regret":[], "Relative Regret":[], "Accuracy":[], "Optimal":[]}
    # eval
    nnet.eval()
    for x, c, w, z in tqdm(dataloader):
        # cuda
        if next(nnet.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = nnet(x)
        # to numpy
        c = c.to("cpu").detach().numpy()
        w = w.to("cpu").detach().numpy()
        z = z.to("cpu").detach().numpy()
        cp = cp.to("cpu").detach().numpy()
        # solve
        for i in range(cp.shape[0]):
            # sol for pred cost
            optmodel.setObj(cp[i])
            wpi, _ = optmodel.solve()
            # obj with true cost
            zpi = np.dot(wpi, c[i])
            # round
            zpi = zpi.round(1)
            zi = z[i,0].round(1)
            # regret
            regret = (zpi - zi).round(1)
            data["Regret"].append(regret)
            data["Relative Regret"].append(regret / zi)
            # accuracy
            data["Accuracy"].append((abs(wpi - w[i]) < 0.5).mean())
            # optimal
            data["Optimal"].append(abs(regret) < 1e-5)
    # dataframe
    df = pd.DataFrame.from_dict(data)
    # print
    time.sleep(1)
    print("Avg Regret: {:.4f}".format(df["Regret"].mean()))
    print("Avg Rel Regret: {:.2f}%".format(df["Relative Regret"].mean()*100))
    print("Path Accuracy: {:.2f}%".format(df["Accuracy"].mean()*100))
    print("Optimality Ratio: {:.2f}%".format(df["Optimal"].mean()*100))
    return df

class hammingLoss(nn.Module):
    def forward(self, wp, w):
        loss = wp * (1.0 - w) + (1.0 - wp) * w
        return loss.mean(dim=0).sum()