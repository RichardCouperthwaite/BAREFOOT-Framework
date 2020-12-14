# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:33:34 2020

@author: richardcouperthwaite
"""

from gpModel import gp_model
from acquisitionFunc import knowledge_gradient
import sys
import numpy as np
import pandas as pd
from pyDOE import lhs
from time import time
from pltEditorTool import plotEditor

def ThreeHumpCamel(x):
    x = x*10 - 5
    if x.shape[0] == 2:
        output = 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2
    else:
        output = 2*x[:,0]**2 - 1.05*x[:,0]**4 + (x[:,0]**6)/6 + x[:,0]*x[:,1] + x[:,1]**2
    return -output

if __name__ == "__main__":
    calc_costs = [5000, 20000, 50000]
    kernels = ['SE', "M32", "M52"]
    sample_count = [10,50,100]
    
    x_plot = []
    y_plot = []
    f_plot = []
    lbls = []
      
    eu_plot = []
    
    qm_min = -1.260351
    qm_max = -1.000325
    lam = 0.1
    
    calc_costs = [5000]
    kernels = ["M52"]
    sample_count = [50]
      
    for ll in range(1):
    
        for kk in range(1):
            
            for mm in range(1):
                iteration_costs = []
                iteration_max_vals = []
                
                calc_cost = calc_costs[mm]
                
                
                for jj in range(3):
                
                    param = ['', kernels[kk], "{}".format(jj)]
                    
                    random_init = pd.read_csv("data/init_data_test.csv",header=None)
                    random_init = np.array(random_init)
                    
                    init_vals = []
                    for ii in range(2):
                        init_vals.append([random_init[jj, ii*2],random_init[jj, (ii*2)+1]])
                    initial_data = np.array(init_vals)
                    
                    
                    x_train = initial_data
                    y_train = ThreeHumpCamel(x_train)
                    
                    cost_list = [0]
                    cost = 0
                    
                    max_out_list = [np.max(y_train)]
                    max_out = np.max(y_train)
                    
                    new_gp = gp_model(x_train, y_train, [0.1,0.1], 10, 0.1, 2, param[1])
                    
                    sCount = sample_count[ll]
                    
                    for ii in range(12):
                        
                        t0 = time()
                        print(ii, max_out)
                        if ii%5 ==0 and ii != 0:
                            sCount += 1 
                        x_test = lhs(2,sCount)
                        
                        mean, cov = new_gp.predict_cov(x_test)
                        
                        nu_star, x_star, NU = knowledge_gradient(sCount, 0.1, mean, cov)
                        
                        new_x = x_test[x_star]
                        new_y = ThreeHumpCamel(new_x)
                        
                        if new_y > max_out:
                            max_out = new_y
                        max_out_list.append(max_out)
                        
                        new_gp.update(new_x, new_y, 0.1, False)
                        
                        cost += (time()-t0) + calc_cost
                        
                        cost_list.append(cost)
                    
                    iteration_costs.append(cost_list)
                    iteration_max_vals.append(max_out_list)
                    
                iteration_cost = np.array(iteration_costs)
                iteration_max_vals = np.array(iteration_max_vals)
                
                mean_cost = np.mean(iteration_cost, axis=0)
                mean_max = np.mean(iteration_max_vals, axis=0)
                std_max = np.std(iteration_max_vals, axis=0)
                
                x_plot.append(mean_cost/1000)
                
                MM_mean = np.abs(mean_max)
                MM_std = std_max
                y_plot.append(np.abs(mean_max))
                
                Z = (-MM_mean-(lam*(MM_std**2))/2)
                new_data_array = -np.exp(-lam*(Z))
                
                eu_plot.append((new_data_array-qm_min)/(qm_max-qm_min))
                f_plot.append(2*std_max)
                lbls.append("{}-{}-{}".format(calc_cost, param[1], sample_count[ll]))
            
            
    plotEditor(x=x_plot, y=y_plot, fill=f_plot, labels=lbls)
    
    plotEditor(x=x_plot, y=eu_plot, labels=lbls)
            
            
            
                    
                
            
        
        
    