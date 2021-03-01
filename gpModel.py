# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:43:08 2020

@author: richardcouperthwaite
"""

from george import kernels, GP
import numpy as np
from copy import deepcopy

class gp_model:
    """
    A class that creates a GP from a given set of input data and hyper-parameters.
    The Kernel can be selected from three separate Kernels.
    """
    def __init__(self, x_train, y_train, l_param, sigma_f, sigma_n, n_dim, kern, mean=0):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.l_param = np.array(l_param)**2
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.mean = mean
        self.n_dim = n_dim
        self.kern = kern
        self.kk = self.create_kernel()
        self.gp = self.create_gp()
        
    def create_kernel(self):
        if self.kern == 'SE':
            return self.sigma_f * kernels.ExpSquaredKernel(self.l_param, ndim=self.n_dim)
        elif self.kern == 'M32':
            return self.sigma_f * kernels.Matern32Kernel(self.l_param, ndim=self.n_dim)
        elif self.kern == 'M52':
            return self.sigma_f * kernels.Matern52Kernel(self.l_param, ndim=self.n_dim)
    
    def create_gp(self):
        gp = GP(kernel=self.kk, mean=self.mean)
        gp.compute(self.x_train, self.sigma_n)
        return gp
    
    def predict_cov(self, x_pred):
        mean, sigma = self.gp.predict(self.y_train, x_pred, kernel = self.kk, return_cov=True, return_var=False)
        return mean, sigma
    
    def predict_var(self, x_pred):
        mean, var = self.gp.predict(self.y_train, x_pred, kernel = self.kk, return_cov=False, return_var=True)
        return mean, var
    
    def update(self, new_x_data, new_y_data, new_y_err, err_per_point):
        self.x_train = np.vstack((self.x_train, new_x_data))
        self.y_train = np.append(self.y_train, new_y_data)
        if err_per_point:
            self.sigma_n = np.append(self.sigma_n, new_y_err)
            
        self.gp = self.create_gp()
        
    def log_likelihood(self):
        return self.gp.log_likelihood(self.y_train, quiet=True)
    
    def get_hyper_params(self):
        curr_params = self.gp.get_parameter_vector()
        params = []
        for i in range(len(curr_params)):
            if i == 0:
                params.append(np.exp(curr_params[i])*self.n_dim)
            else:
                params.append(np.sqrt(np.exp(curr_params[i])))
        return np.array(params)
        
    def hp_optimize(self, meth="L-BFGS-B", update=False):
        import scipy.optimize as op
        gp = deepcopy(self)
        p0 = gp.gp.get_parameter_vector()
        def nll(p):
            gp.gp.set_parameter_vector(p)
            ll = gp.log_likelihood()
            return -ll if np.isfinite(ll) else 1e25
        
        def grad_nll(p):
            gp.gp.set_parameter_vector(p)
            return -gp.gp.grad_log_likelihood(self.y_train, quiet=True)
                
        results = op.minimize(nll, p0, jac=grad_nll, method=meth)
        if update:
            # automatically update the hyper-parameters, the required input for
            # the set_parameter_vector command is the log of the hyper-parameters
            self.gp.set_parameter_vector(results.x)
            self.gp.compute(self.x_train, self.sigma_n)
        # The results are the log of the hyper-parameters, so return the
        # exponential of the results.
        return np.exp(results.x)