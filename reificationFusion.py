# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:48:44 2020

@author: richardcouperthwaite
"""

import numpy as np
from gpModel import gp_model
import matplotlib.pyplot as plt

def reification(y, sig):
    """
    This function is coded to enable the reification of any number of models.
    """
    mean_fused = []
    var_fused = []
    
    rtest = []
        
    rho_bar = []
    for i in range(len(y)-1):
        for j in range(len(y)-i-1):
#            y[j+i][np.where((np.abs(y[i]-0)<eps)*(np.abs(y[j+i]-0)<eps))] = 10
            
            rho1 = np.divide(np.sqrt(sig[i]), np.sqrt((y[i]-y[j+i+1])**2 + sig[i]))
            rtest.append(rho1)
            rho2 = np.divide(np.sqrt(sig[j+i+1]), np.sqrt((y[j+i+1]-y[i])**2 + sig[j+i+1]))
            rtest.append(rho2)
            rho_bar_ij = np.divide(sig[j+i+1], (sig[i]+sig[j+i+1]))*rho1 + np.divide(sig[i], (sig[i]+sig[j+i+1]))*rho2
            rho_bar_ij[np.where(rho_bar_ij>0.99)] = 0.99
            rho_bar.append(rho_bar_ij)
            
    mm = rho_bar[0].shape[0]
            
    sigma = np.zeros((len(y), len(y)))
    
    for i in range(mm):
        for j in range(len(y)):
            for k in range(len(y)-j):
                jj = j
                kk = k+j
                if jj == kk:
                    sigma[jj,kk] = sig[jj][i]
                else:
                    sigma[jj,kk] = rho_bar[kk+jj-1][i]*np.sqrt(sig[jj][i]*sig[kk][i])
                    sigma[kk,jj] = rho_bar[kk+jj-1][i]*np.sqrt(sig[jj][i]*sig[kk][i])

        alpha = np.linalg.inv(sigma)
        w = np.sum(alpha,1)/np.sum(alpha)
        mu = y[0][i]
        for j in range(len(y)-1):
            mu = np.hstack((mu,y[j+1][i]))
        mean = np.sum(w@mu)
        
        mean_fused.append(mean)
        var_fused.append(1/np.sum(alpha))
        
    return np.array(mean_fused), np.array(var_fused)

class model_reification():
    def __init__(self, x_train, y_train, l_param, sigma_f, sigma_n, model_mean,
                 model_std, l_param_err, sigma_f_err, sigma_n_err, 
                 x_true, y_true, num_models, num_dim, kernel):
        self.x_train = x_train
        self.y_train = y_train
        self.model_mean = model_mean
        self.model_std = model_std
        self.model_hp = {"l": np.array(l_param),
                         "sf": np.array(sigma_f),
                         "sn": np.array(sigma_n)}
        self.err_mean = []
        self.err_std = []
        self.err_model_hp = {"l": np.array(l_param_err),
                             "sf": np.array(sigma_f_err),
                             "sn": np.array(sigma_n_err)}
        self.x_true = x_true
        self.y_true = y_true
        self.num_models = num_models
        self.num_dim = num_dim
        self.kernel = kernel
        self.gp_models = self.create_gps()
        self.gp_err_models = self.create_error_models()
        self.fused_GP = ''
        self.fused_y_mean = ''
        self.fused_y_std = ''
        
    def create_gps(self):
        """
        GPs need to be created for each of the lower dimension information sources
        as used in the reification method. These can be multi-dimensional models.
        As a result, the x_train and y_train data needs to be added to the class
        as a list of numpy arrays.
        """
        gp_models = []
        for i in range(self.num_models):
            new_model = gp_model(self.x_train[i], 
                                 (self.y_train[i]-self.model_mean[i])/self.model_std[i], 
                                 self.model_hp["l"][i], 
                                 self.model_hp["sf"][i], 
                                 self.model_hp["sn"][i], 
                                 self.num_dim, 
                                 self.kernel)
            gp_models.append(new_model)
        return gp_models
    
    def create_error_models(self):
        """
        In order to calculate the total error of the individual GPs an error
        model is created for each GP. The inputs to this are the error between
        the individual GP predictions and the truth value at all available truth
        data points. The prior_error value is subtracted from the difference to
        ensure that the points are centred around 0.
        """
        gp_error_models = []
        for i in range(self.num_models):
            gpmodel_mean, gpmodel_var = self.gp_models[i].predict_var(self.x_true)
            gpmodel_mean = gpmodel_mean * self.model_std[i] + self.model_mean[i]
            error = np.abs(self.y_true-gpmodel_mean)
            self.err_mean.append(np.mean(error))
            self.err_std.append(np.std(error))
            if self.err_std[i] == 0:
                self.err_std[i] = 1
            new_model = gp_model(self.x_true, 
                                 (error-self.err_mean[i])/self.err_std[i], 
                                 self.err_model_hp["l"][i], 
                                 self.err_model_hp["sf"][i], 
                                 self.err_model_hp["sn"][i], 
                                 self.num_dim, 
                                 self.kernel)
            gp_error_models.append(new_model)
        return gp_error_models
    
    def create_fused_GP(self, x_test, l_param, sigma_f, sigma_n, kernel):
        model_mean = []
        model_var = []
        for i in range(len(self.gp_models)):
            m_mean, m_var = self.gp_models[i].predict_var(x_test)
            m_mean = m_mean * self.model_std[i] + self.model_mean[i]
            m_var = m_var * (self.model_std[i] ** 2)
            model_mean.append(m_mean)
            err_mean, err_var = self.gp_err_models[i].predict_var(x_test)
            err_mean = err_mean * self.err_std[i] + self.err_mean[i]
            model_var.append((err_mean)**2 + m_var)
        fused_mean, fused_var = reification(model_mean, model_var)
        self.fused_y_mean = np.mean(fused_mean[0:400:12])
        self.fused_y_std = np.std(fused_mean[0:400:12])
        if self.fused_y_std == 0:
            self.fused_y_std = 1
        fused_mean = (fused_mean - self.fused_y_mean)/self.fused_y_std
        self.fused_GP = gp_model(x_test[0:400:12], 
                                 fused_mean[0:400:12], 
                                 l_param, 
                                 sigma_f, 
                                 abs(fused_var[0:400:12])**(0.5), 
                                 self.num_dim, 
                                 kernel)
        return self.fused_GP
        
    def update_GP(self, new_x, new_y, model_index):
        
        # new_x = np.expand_dims(new_x, axis=0)
        # print(new_x.shape)
        # print(self.x_train[model_index].shape)
        # self.x_train[model_index] = np.append(self.x_train[model_index], new_x)
        # self.y_train[model_index] = np.append(self.y_train[model_index], new_y)
        self.x_train[model_index] = np.vstack((self.x_train[model_index], new_x))
        self.y_train[model_index] = np.append(self.y_train[model_index], new_y)
        self.gp_models[model_index].update(new_x, new_y, self.model_hp['sn'][model_index], False)
    
    def update_truth(self, new_x, new_y):
        self.x_true = np.vstack((self.x_true, new_x))
        self.y_true = np.append(self.y_true, new_y)
        self.gp_err_models = self.create_error_models()
        
    def predict_low_order(self, x_predict, index):
        gpmodel_mean, gpmodel_var = self.gp_models[index].predict_var(x_predict)
        gpmodel_mean = gpmodel_mean * self.model_std[index] + self.model_mean[index]
        gpmodel_var = gpmodel_var * (self.model_std[index]**2)
        return gpmodel_mean, gpmodel_var
    
    def predict_fused_GP(self, x_predict):
        gpmodel_mean, gpmodel_var = self.fused_GP.predict_var(x_predict)
        gpmodel_mean = gpmodel_mean * self.fused_y_std + self.fused_y_mean
        gpmodel_var = gpmodel_var * (self.fused_y_std**2)
        return gpmodel_mean, np.diag(gpmodel_var)
    
    def plot_models(self, tc_gp, iteration, file_dir):
        for i in range(len(self.gp_models)):
            tc_out = tc_gp.predict(self.x_train[i])
            try:
                plt.figure(1)
                plt.scatter(tc_out[:,0], self.y_train[i])
                plt.savefig("results/{}/figures/model{}_{}.png".format(file_dir,i,iteration))
                plt.close(1)
            except:
                print("model{}_{} Plot Failed".format(i,iteration))
            
        tc_out = tc_gp.predict(self.x_true)
        try:
            plt.figure(1)
            plt.scatter(tc_out[:,0], self.y_true)
            plt.savefig("results/{}/figures/RVE_{}.png".format(file_dir,iteration))
            plt.close(1)
        except:
            print("RVE_{} Plot Failed".format(iteration))