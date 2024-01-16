import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer 
from src.dys_opt_net import DYS_opt_net
from pyepo.model.grb import shortestPathModel