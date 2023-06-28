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