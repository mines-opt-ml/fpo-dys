
import numpy as np
import torch

from pyepo import EPO

def accuracy(predmodel, optmodel, dataloader):
    """
    A function to evaluate model performance with accuracy.
    Writtne by Daniel McKenzie in the style of PyEPO.

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
    # load data
    for data in dataloader:
        x, c, w, z = data
        # cuda
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        with torch.no_grad(): # no grad
            cp = predmodel(x).to("cpu").detach().numpy()
        # solve
        for j in range(cp.shape[0]):
            # accumulate loss
            loss += calAccuracy(optmodel, cp[j], w[j].to("cpu").detach().numpy())
    # turn back train mode
    predmodel.train()
    # normalized
    return loss /cp.shape[0] # divide by batch size


def calAccuracy(optmodel, pred_cost, true_sol):
    """
    A function to calculate normalized true regret for a batch

    Args:
        optmodel (optModel): optimization model
        pred_cost (torch.tensor): predicted costs
        true_sol (torch.tensor): true solution

    Returns:
        acc: 1 if true_sol matches that computed using pred_cost to within a small tolerance.
        
    """
    # opt sol for pred cost
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    if np.linalg.norm(sol - true_sol)< 1e-4:
        acc = 1
    else:
        acc = 0
    return acc
