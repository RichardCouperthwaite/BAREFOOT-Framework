# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:34:06 2020

@author: Richard Couperthwaite
"""

from barefoot import batch_optimization
import sys
import numpy as np
import pandas as pd

def hartmann6(x):
    x = np.expand_dims(x,1)
    x_matrix = np.tile(x, (1,4,1))
    samples = x.shape[0]
    
    aA = np.tile(np.array([[10,3,17,3.5,1.7,8],
                            [0.05,10,17,0.1,8,14],
                            [3,3.5,1.7,10,17,8],
                            [17,8,0.05,10,0.1,14]]), (samples,1,1))
    pP = np.tile(np.array([[1312,1696,5569,124,8283,5886],
                            [2329,4135,8307,3736,1004,9991],
                            [2348,1451,3522,2883,3047,6650],
                            [4047,8828,8732,5743,1091,381]]), (samples,1,1))
    pP = pP*1e-4
    alph = np.tile(np.array([1.0,1.2,3.0,3.2]), (samples,1))
    
    exp_sum = aA * ((x_matrix - pP)**2)
    
    return (1/1.94)*(2.58+np.sum(np.exp(-np.sum(exp_sum,axis=2))*alph, axis=1))

def hartmann6_LO1(x):
    x = np.expand_dims(x,1)
    x_matrix = np.tile(x, (1,4,1))
    samples = x.shape[0]
    
    factor1 = 0.45
    factor2 = 0.32
    factor3 = -1.94
    factor4 = -1.32
        
    aA = factor1*np.tile(np.array([[10,3,17,3.5,1.7,8],
                           [0.05,10,17,0.1,8,14],
                           [3,3.5,1.7,10,17,8],
                           [17,8,0.05,10,0.1,14]]), (samples,1,1))
    pP = factor2*np.tile(np.array([[1312,1696,5569,124,8283,5886],
                           [2329,4135,8307,3736,1004,9991],
                           [2348,1451,3522,2883,3047,6650],
                           [4047,8828,8732,5743,1091,381]]), (samples,1,1))
    pP = pP*1e-4
    alph = np.tile(np.array([factor3*1.0,1.2,factor4*3.0,3.2]), (samples,1))
    
    exp_sum = aA * ((x_matrix - pP)**2)
    
    return (1/1.94)*(2.58+np.sum(np.exp(-np.sum(exp_sum,axis=2))*alph, axis=1))

def hartmann6_LO2(x):
    x = np.expand_dims(x,1)
    x_matrix = np.tile(x, (1,4,1))
    samples = x.shape[0]
    
    factor1 = 1.37
    factor2 = 0.57
    factor3 = -1.52
    factor4 = -0.5
    
    aA = factor1*np.tile(np.array([[10,3,17,3.5,1.7,8],
                           [0.05,10,17,0.1,8,14],
                           [3,3.5,1.7,10,17,8],
                           [17,8,0.05,10,0.1,14]]), (samples,1,1))
    pP = factor2*np.tile(np.array([[1312,1696,5569,124,8283,5886],
                           [2329,4135,8307,3736,1004,9991],
                           [2348,1451,3522,2883,3047,6650],
                           [4047,8828,8732,5743,1091,381]]), (samples,1,1))
    pP = pP*1e-4
    alph = np.tile(np.array([factor3*1.0,1.2,factor4*3.0,3.2]), (samples,1))
    
    exp_sum = aA * ((x_matrix - pP)**2)
    
    return (1/1.94)*(2.58+np.sum(np.exp(-np.sum(exp_sum,axis=2))*alph, axis=1))

def hartmann6_LO3(x):
    x = np.expand_dims(x,1)
    x_matrix = np.tile(x, (1,4,1))
    samples = x.shape[0]
    
    factor1 = 1.86
    factor2 = 1.7
    factor3 = -0.15
    factor4 = -0.35
    
    aA = factor1*np.tile(np.array([[10,3,17,3.5,1.7,8],
                           [0.05,10,17,0.1,8,14],
                           [3,3.5,1.7,10,17,8],
                           [17,8,0.05,10,0.1,14]]), (samples,1,1))
    pP = factor2*np.tile(np.array([[1312,1696,5569,124,8283,5886],
                           [2329,4135,8307,3736,1004,9991],
                           [2348,1451,3522,2883,3047,6650],
                           [4047,8828,8732,5743,1091,381]]), (samples,1,1))
    pP = pP*1e-4
    alph = np.tile(np.array([factor3*1.0,1.2,factor4*3.0,3.2]), (samples,1))
    
    exp_sum = aA * ((x_matrix - pP)**2)
    
    return (1/1.94)*(2.58+np.sum(np.exp(-np.sum(exp_sum,axis=2))*alph, axis=1))

def hartmann6_LO4(x):
    x = np.expand_dims(x,1)
    x_matrix = np.tile(x, (1,4,1))
    samples = x.shape[0]
    
    factor1 = 0.74
    factor2 = 1.07
    factor3 = -0.89
    factor4 = -1.28
    
    aA = factor1*np.tile(np.array([[10,3,17,3.5,1.7,8],
                           [0.05,10,17,0.1,8,14],
                           [3,3.5,1.7,10,17,8],
                           [17,8,0.05,10,0.1,14]]), (samples,1,1))
    pP = factor2*np.tile(np.array([[1312,1696,5569,124,8283,5886],
                           [2329,4135,8307,3736,1004,9991],
                           [2348,1451,3522,2883,3047,6650],
                           [4047,8828,8732,5743,1091,381]]), (samples,1,1))
    pP = pP*1e-4
    alph = np.tile(np.array([factor3*1.0,1.2,factor4*3.0,3.2]), (samples,1))
    
    exp_sum = aA * ((x_matrix - pP)**2)
    
    return (1/1.94)*(2.58+np.sum(np.exp(-np.sum(exp_sum,axis=2))*alph, axis=1))

def hartmann6_LO5(x):
    x = np.expand_dims(x,1)
    x_matrix = np.tile(x, (1,4,1))
    samples = x.shape[0]
    
    factor1 = 0.74
    factor2 = 0.92
    factor3 = -0.9
    factor4 = -1.58
    
    aA = factor1*np.tile(np.array([[10,3,17,3.5,1.7,8],
                           [0.05,10,17,0.1,8,14],
                           [3,3.5,1.7,10,17,8],
                           [17,8,0.05,10,0.1,14]]), (samples,1,1))
    pP = factor2*np.tile(np.array([[1312,1696,5569,124,8283,5886],
                           [2329,4135,8307,3736,1004,9991],
                           [2348,1451,3522,2883,3047,6650],
                           [4047,8828,8732,5743,1091,381]]), (samples,1,1))
    pP = pP*1e-4
    alph = np.tile(np.array([factor3*1.0,1.2,factor4*3.0,3.2]), (samples,1))
    
    exp_sum = aA * ((x_matrix - pP)**2)
    
    return (1/1.94)*(2.58+np.sum(np.exp(-np.sum(exp_sum,axis=2))*alph, axis=1))

def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape

if __name__ == "__main__":
    param = sys.argv

    if len(param) != 21:
        param = ['', 'M52', 2, 5, 10, 2, 2, 500000, 1000000,
                  "hartmann_test", 6, 10, 0, 2, 0.000001, 2, 5000, 0.0001, 3, False, False]
    
    all_models = [hartmann6, hartmann6_LO1,
                  hartmann6_LO2, hartmann6_LO3,
                  hartmann6_LO4, hartmann6_LO5]
    all_model_cost_adj = [0.9, 1.1, 5, 10, 6]
    all_model_params = [[[0.1,0.1,0.1,0.1,0.1,0.1],1,0,1],
                        [[0.1,0.1,0.1,0.1,0.1,0.1],1,0,1],
                        [[0.1,0.1,0.1,0.1,0.1,0.1],1,0,1],
                        [[0.1,0.1,0.1,0.1,0.1,0.1],1,0,1],
                        [[0.1,0.1,0.1,0.1,0.1,0.1],1,0,1]]
    
    kernel = param[1] #'M52'      # define the kernel to be used as the Matern 5/2 kernel
    iter_count = int(param[2]) #201     # define the number of iterations
    sample_count = int(param[3]) #50   # define the number of samples to test on (modified by classifier)
    hp_count = int(param[4]) #500       # define the number of hyper-parameter sets
    num_medoids = int(param[5]) #2     # define the number of medoid clusters to use
    tm_iter = int(param[6]) #10        # define the number of iterations between each RVE call
    total_budget = int(param[7])
    tm_budget = int(param[8])
    exp_name = param[9]
    
    ndim = int(param[10])
    fused_points = int(param[11])
    init_index = int(param[12])
    num_init = int(param[13])
    
    low_bound = float(param[14])
    upper_bound = float(param[15])
    
    models = [all_models[0]]
    model_param = {'model_l': [],
                    'model_sf': [],
                    'model_sn': [],
                    'means': [],
                    'std': [],
                    'err_l': [],
                    'err_sf': [],
                    'err_sn': [],
                    'costs': [],
                    'Truth Cost': float(param[16])}

    base_model_cost = float(param[17])

    for ii in range(int(param[18])):
        models.append(all_models[ii+1])
        model_param['model_l'].append(all_model_params[ii][0])
        model_param['model_sf'].append(all_model_params[ii][1])
        model_param['model_sn'].append(0.01)
        model_param['means'].append(all_model_params[ii][2])
        model_param['std'].append(all_model_params[ii][3])
        model_param['err_l'].append(0.1)
        model_param['err_sf'].append(10)
        model_param['err_sn'].append(0.01)
        model_param['costs'].append(base_model_cost*all_model_cost_adj[ii])
        
    
    random_init = pd.read_csv("data/init_data.csv",header=None)
    random_init = np.array(random_init)
    
    # define the initial data point, this will be the same for all models.
    init_vals = []
    for ii in range(num_init):
        init_vals.append(random_init[init_index, ii*6:ii*6+6])
    initial_data = np.array(init_vals) 
    
    batch_optimization(param, ndim, fused_points, initial_data, models, 
                        low_bound, upper_bound, model_param, [int(param[19]), int(param[20])])
    