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
    ''' Abstract implementation of a Davis-Yin Splitting (DYS) layer in a neural network.

        Note:
            The singular value decomposition of the matrix $\mathsf{A}$ is used for the
            projection onto the subspace of all $\mathsf{x}$ such that $\mathsf{Ax=b}$.

        Args:
            A (tensor):      Matrix for linear system
            b (tensor):      Measurement vector for linear system
            device (string): Device on which to perform computations
            alpha (float):   Step size for DYS updates
     
    '''
    def __init__(self, A, b, device='mps', alpha=0.05):
        super().__init__()
        self.device = device
        self.b = b
        self.alpha = alpha*torch.ones(1, device=self.device)
        self.A = A
        self.n1 = A.shape[0]  # Number of rows of A
        self.n2 = A.shape[1]  # Number of columns of A

        U, s, VT = torch.linalg.svd(self.A, full_matrices=False)
        self.s_inv = torch.tensor([1/sing if sing >=1e-6 else 0 for sing in s]).to(self.device)
        self.V = torch.t(VT).to(self.device)
        self.UT = torch.t(U).to(self.device)

    def _project_C1(self, x):
        ''' Projection to the non-negative orthant.

        Args:
            x (tensor): point in Euclidean space

        Returns:
            Px (tensor): projection of $\mathsf{x}$ onto nonnegative orthant
        
        '''
        Px = torch.clamp(x, min=0)
        return Px

    def _project_C2(self, z):
      ''' Projection to the subspace of all $\mathsf{x}$ such that $\mathsf{Ax=b}$.

        Note:
            The singular value decomposition (SVD) representation
            of the matrix $\mathsf{A}$ is used to efficiently compute
            the projection.
            
        Args:
            z (tensor): point in Euclidean space

        Returns:
            Pz (tensor): projection onto subspace $\mathsf{\{z : Ax = b\}}$
         
      '''
      res = self.A.matmul(z.permute(1,0)) - self.b.view(-1,1)
      temp = self.V.matmul(self.s_inv.view(-1,1)*self.UT.matmul(res)).permute(1,0)
      Pz = z - temp
      return Pz

    @abstractmethod
    def F(self, z, w):
        ''' Gradient of objective function. Must be defined for each problem type.
       
            Note:
                The parameters of $\mathsf{F}$ are stored in $\mathsf{w}$.

            Args:
                z (tensor): point in Euclidean space
                w (tensor): Parameters defining function and its gradient
        '''
        pass

    @abstractmethod
    def _data_space_forward(self, d):
      ''' Specify the map from context d to parameters of F.
      '''
      pass

    @abstractmethod
    def test_time_forward(self, d):
       '''
       Specify test time behaviour, e.g. use a combinatorial solver on the forward pass.
       '''
       pass


    def _apply_DYS(self, z, w): 
        ''' Apply a single update step from Davis-Yin Splitting. 
            
            Args:
                z (tensor): Point in Euclidean space
                w (tensor): Parameters defining function and its gradient

            Returns:
                z (tensor): Updated estimate of solution
        '''
        x = self._project_C1(z)
        y = self._project_C2(2.0 * x - z - self.alpha*self.F(z, w))
        z = z - x + y
        return z


    def _train_time_forward(self, d, eps=1.0e-2, max_depth=int(1e4), 
                depth_warning=True): 
        ''' Default forward behaviour during training.

            Args:
                d (tensor):           Contextual data
                eps (float);          Stopping criterion threshold
                max_depth (int):      Maximum number of DYS updates
                depth_warning (bool): Boolean for whether to print warning message when max depth reached
            
            Returns:
                z (tensor): P+O Inference
        '''
        with torch.no_grad():
            w = self._data_space_forward(d)
            self.depth = 0.0

            z = torch.rand((self.n2), device=self.device)
            z_prev = z.clone()      
          
            all_samp_conv = False
            while not all_samp_conv and self.depth < max_depth:
                z_prev = z.clone()   
                z = self._apply_DYS(z, w)
                diff_norm = torch.norm(z - z_prev) 
                diff_norm = torch.norm( diff_norm) 
                diff_norm = torch.max( diff_norm ) # take norm along the latter two dimensions then max
                self.depth += 1.0
                all_samp_conv = diff_norm <= eps
            
        if self.depth >= max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")
        if self.training:
            w = self._data_space_forward(d)
            z = self._apply_DYS(z.detach(), w)
            return self._project_C1(z)
        else:
            return self._project_C1(z).detach()

    
    def forward(self, d, eps=1.0e-2, max_depth=int(1e4),
                depth_warning=True):
        ''' Forward propagation of DYS-net.
        
            Note:
                A switch is included for using different behaviour at test/deployment. 

            Args:
                d (tensor):           Contextual data
                eps (float);          Stopping criterion threshold
                max_depth (int):      Maximum number of DYS updates
                depth_warning (bool): Boolean for whether to print warning message when max depth reached
            
            Returns:
                z (tensor): P+O Inference        
        '''
        if not self.training:
          return self.test_time_forward(d)

        return self._train_time_forward(d, eps, max_depth, depth_warning)
