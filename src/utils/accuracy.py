
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
import numpy.random as random

from pyepo import EPO

def accuracy(predmodel, optmodel, dataloader):
    """
    A function to evaluate model performance with accuracy.
    Written by Daniel McKenzie in the style of PyEPO.

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): an PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true regret loss
    """
    # evaluate
    predmodel.eval()
    loss = 0
    optsum = 0
    num_batches = 0
    # load data
    for data in dataloader:
        x, c, w, z = data
        batch_loss = 0
        num_batches += 1
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        with torch.no_grad(): # no grad
            cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            batch_loss += calAccuracy(optmodel, cp[j], w[j].to("cpu").detach().numpy(),c[j].to("cpu").detach().numpy())
        loss += batch_loss/cp.shape[0]
    # turn back train mode
    predmodel.train()
    print(f'num batches is {num_batches}')
    # normalized
    return loss/num_batches # divide by number of batches


def calAccuracy(optmodel, pred_cost, true_sol, true_cost):
    """
    A function to calculate accuracy, where the prediction is considered "accurate" if
    it yields almost the same objective function value as the true solution.

    Args:
        optmodel (optModel): optimization model
        pred_cost (torch.tensor): predicted cost vector
        true_sol (torch.tensor): true solution
        true_cost: ground truth cost vector

    Returns:
        acc: 1 if predicted solution yields almost the same objective function value as true_sol.
        
    """
    # opt sol for pred cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    # ## plotting utility
    # if idx % 25 == 0:
    #     fig, axes = plt.subplots(1,2)
    #     cax1 = axes[0].matshow(true_sol.reshape(12,12))
    #     axes[0].set_title('True')

    #     cax2 = axes[1].matshow(sol.reshape(12,12))
    #     axes[1].set_title('Predicted')
    #     rand_number = random.randint(1,1000)
    #     plt.savefig(os.path.join('./src/warcraft/imgs/','comparing_outputs_batch_id='+str(idx)+".png"))
    if np.dot(true_cost, sol - true_sol) < 1e-3: # np.linalg.norm(sol - true_sol)< 1e-4:
        acc = 1
    else:
        acc = 0
    return acc
