#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:19:54 2022

@author: danielmckenzie

The DYS Layer. Pure gold
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DYS_opt_net(nn.Module, ABC):
    def __init__(self, A, b, alpha=0.05, device='cuda:0'):
        super().__init__()
        # self.b = b.to(device) # assumes b has shape n (not nx1)
        self.b = b.to(device)
        self.alpha = alpha*torch.ones(1, device=device)
        self.A = A.to(device)
        # self.A = self.A.cuda()
        # self.A = A
        self.n1 = A.shape[0]  # Number of rows of A
        self.n2 = A.shape[1]  # Number of columns of A
        self.device = device

        U, s, VT = torch.linalg.svd(self.A, full_matrices=False)
        self.s_inv = torch.tensor([1/sing if sing >=1e-6 else 0 for sing in s]).to(device)
        self.V = torch.t(VT).to(device)
        self.UT = torch.t(U).to(device)

    def project_C1(self, x):
        '''
        Projection to the non-negative orthant.
        '''
        return torch.clamp(x, min=0, max=1)

    def project_C2(self, z):
      '''
      Projection to the subspace Ax=b.
      '''
      res = self.A.matmul(z.permute(1,0)) - self.b.view(-1,1)
      temp = self.V.matmul(self.s_inv.view(-1,1)*self.UT.matmul(res)).permute(1,0)
      Pz = z - temp
      return Pz

    @abstractmethod
    def F(self, z, w):
        '''
        Gradient of objective function. Must be defined for each problem type.
        Note the parameters of F are stored in w.
        '''
        pass

    @abstractmethod
    def data_space_forward(self, d):
      '''
      Specify the map from context d to parameters of F.
      '''
      pass


    def apply_DYS(self, z, w): 
      """
        Davis-Yin Splitting. 
      """

      x = self.project_C1(z)
      y = self.project_C2(2.0*x - z - self.alpha*self.F(z, w))
      z = z - x + y

      return z


    def forward(self, d, eps=1.0e-2, max_depth=int(1e4), 
                depth_warning=False): 
      """
      w are the parameters. To be passed to the operator F.
      """

      with torch.no_grad():
          w = self.data_space_forward(d)
          self.depth = 0.0

          z = torch.rand((self.n2), device=self.device)
          z_prev = z.clone()      
          
          all_samp_conv = False
          while not all_samp_conv and self.depth < max_depth:
              z_prev = z.clone()   
              z = self.apply_DYS(z, w)
              diff_norm = torch.norm(z - z_prev) 
              diff_norm = torch.norm( diff_norm) 
              diff_norm = torch.max( diff_norm ) # take norm along the latter two dimensions then max
              self.depth += 1.0
              all_samp_conv = diff_norm <= eps
            
      if self.depth >= max_depth and depth_warning:
          print("\nWarning: Max Depth Reached - Break Forward Loop\n")

      attach_gradients = self.training
      if attach_gradients:
          w = self.data_space_forward(d)
          z = self.apply_DYS(z.detach(), w)
          return self.project_C1(z)
      else:
          return self.project_C1(z).detach()
