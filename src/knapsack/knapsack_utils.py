import torch.nn as nn
import torch

# Make eval mode that rounds before computing cost.
class RegretLoss(nn.Module):
    def __init__(self, num_item, num_constraints, device):
        super(RegretLoss, self).__init__()
        self.device = device
        self.num_items = num_item  # number of items, i.e. non-dummy variables
        self.num_constraints = num_constraints # number of constraints, i.e. dummy variables
    
    def forward(self, w_true, x_pred, x_opt, opt_value, eval_mode=False):
        '''
          d is (batch of) contexts, w_true is (batch of) true 
          cost vectors.
        '''
        if eval_mode:
            regret = opt_value - (w_true*torch.round(x_pred)).sum(dim=-1)
            regret = torch.mean(regret)/torch.mean(opt_value) #normalized regret
        else:
            regret = opt_value - (w_true*x_pred).sum(dim=-1)
            regret = torch.mean(regret)
        return regret
    
## Utility for computing validation or test loss.
def Compute_Test_Loss(net,loader, model_type, metric, num_knapsack, num_item, device):
    net.eval()
    loss = 0
    for d_batch, w_batch, opt_sol, opt_value in loader:
        d_batch = d_batch.to(device)
        w_batch = w_batch.to(device)
        opt_sol = opt_sol.to(device)
        opt_value = opt_value.to(device)
        predicted = net(d_batch)
        if model_type == "DYS" or model_type == "DYS-Regret":
            loss += metric(w_batch, predicted[:,:-(num_knapsack + num_item)], opt_sol, opt_value, eval_mode=True).item()
        else: 
            loss += metric(w_batch, predicted, opt_sol, opt_value, eval_mode=True).item()
    return loss
        