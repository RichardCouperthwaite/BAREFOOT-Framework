# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:12:49 2020

@author: richardcouperthwaite
"""

from BAREFOOT import batch_optimization, batch_optimization_v2
import sys
import numpy as np
import pandas as pd


def ThreeHumpCamel(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2
    else:
        output = 2*x[:,0]**2 - 1.05*x[:,0]**4 + (x[:,0]**6)/6 + x[:,0]*x[:,1] + x[:,1]**2
    return -output

def ThreeHumpCamel_LO1(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 1.05*(x[0]-0.5)**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2
    else:
        output = 1.05*(x[:,0]-0.5)**4 + (x[:,0]**6)/6 + x[:,0]*x[:,1] + x[:,1]**2
    return -output

def ThreeHumpCamel_LO2(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 2*(x[0]+0.5)**2 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2
    else:
        output = 2*(x[:,0]+0.5)**2 + (x[:,0]**6)/6 + x[:,0]*x[:,1] + x[:,1]**2
    return -output

def ThreeHumpCamel_LO3(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 2*(x[0]*0.5)**2 - 1.05*x[0]**4 + x[0]*x[1] + x[1]**2
    else:
        output = 2*(x[:,0]*0.5)**2 - 1.05*x[:,0]**4 + x[:,0]*x[:,1] + x[:,1]**2
    return -output

def ThreeHumpCamel_LO4(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 2*(x[0]*2)**2 - 1.05*x[1]**4 + (x[0]**6)/6 + x[1]**2
    else:
        output = 2*(x[:,0]*2)**2 - 1.05*x[:,1]**4 + (x[:,0]**6)/6 + x[:,1]**2
    return -output

def ThreeHumpCamel_LO5(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 2*(x[1])**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1]
    else:
        output = 2*(x[:,1])**2 - 1.05*x[:,0]**4 + (x[:,0]**6)/6 + x[:,0]*x[:,1]
    return -output

if __name__ == "__main__":
    print("Code Started Successfully")
    
    if input("Continue with calculation?     ") == "y":

        param = sys.argv

        if len(param) != 21:
            param = ['', 'M52', 4, 10, 30, 2, 2, 500000, 1000000,
                      "Test_20", 2, 10, 0, 2, 0.000001, 2, 5000, 0.0001, 3, False, False]
        
        all_models = [ThreeHumpCamel, ThreeHumpCamel_LO1,
                      ThreeHumpCamel_LO2, ThreeHumpCamel_LO3,
                      ThreeHumpCamel_LO4, ThreeHumpCamel_LO5]
        all_model_cost_adj = [0.9, 1.1, 5, 10, 6]
        all_model_params = [[[0.1,0.1],1,-662.349,992.766],
                            [[0.1,0.1],1,-449.358,725.333],
                            [[0.1,0.1],1,122.590,185.487],
                            [[0.1,0.1],1,-357.254,789.272],
                            [[0.1,0.1],1,-303.636,525.854]]
        
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
            init_vals.append([random_init[init_index, ii*2],random_init[init_index, (ii*2)+1]])
        initial_data = np.array(init_vals)  
        
        batch_optimization_v2(param, ndim, fused_points, initial_data, models, 
                            low_bound, upper_bound, model_param, [int(param[19]), int(param[20])])