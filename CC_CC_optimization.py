# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:13:50 2020

@author: richardcouperthwaite
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from functions import gp_model, reification, knowledge_gradient
from reduced_order_models import isostrain_IS, isostress_IS, isowork_IS
from time import time
from copy import deepcopy
from pyDOE import lhs
from kmedoids import kMedoids
import os
import sys
import multiprocessing
from time import sleep
import datetime as dt

class RVE_GP():
    def __init__(self):
        self.mean = 0
        self.std = 0
        self.gp = 0
        self.setup()
        
    def setup(self):
        data = pd.read_excel('data/rve_data.xlsx')
        data.iloc[:,0] = (data.iloc[:,0]-650)/200
        data.iloc[:,2] = data.iloc[:,2]/3
        data.iloc[:,3] = data.iloc[:,3]/2
        self.mean = np.mean(data.iloc[:,5])
        self.std = np.std(data.iloc[:,5])
        data.iloc[:,5] = (data.iloc[:,5]-self.mean)/self.std
        self.gp = gp_model(data.iloc[:,0:4], data.iloc[:,5], np.array([0.12274117, 0.08612411, 0.65729583, 0.23342798]), 0.16578065, 0.1, 4, 'SE')

    def predict(self, x_predict):
        if len(x_predict.shape) == 1:
            x_predict = np.expand_dims(x_predict, axis=0)
        x = np.ones((x_predict.shape[0],4))        
        x[:,0] = (x_predict[:,0]-650)/200 #Temperature
        x[:,1] = x_predict[:,1]           #wt% C
        x[:,2] = x[:,2]*0.283/2           #wt% Si
        x[:,3] = x[:,3]*0.328/3           #wt% Mn
        mean, var = self.gp.predict_var(x)
        
        return mean*self.std + self.mean
    
    def test_fit(self):
        data = pd.read_excel('data/rve_data.xlsx')
        data_1 = deepcopy(data)
        data.iloc[:,0] = (data.iloc[:,0]-650)/200
        data.iloc[:,2] = data.iloc[:,2]/3
        data.iloc[:,3] = data.iloc[:,3]/2
        
        test_data = [[],[],[],[],[],[],[],[],[],[]]
        train_data = [[],[],[],[],[],[],[],[],[],[]]
        count = 1
        while count <= 1500:
            new_num = np.random.randint(0,1522)
            if (new_num not in test_data[0]) and (len(test_data[0])<150):
                test_data[0].append(new_num)
                count += 1
            elif (new_num not in test_data[1]) and (len(test_data[1])<150):
                test_data[1].append(new_num)
                count += 1
            elif (new_num not in test_data[2]) and (len(test_data[2])<150):
                test_data[2].append(new_num)
                count += 1
            elif (new_num not in test_data[3]) and (len(test_data[3])<150):
                test_data[3].append(new_num)
                count += 1
            elif (new_num not in test_data[4]) and (len(test_data[4])<150):
                test_data[4].append(new_num)
                count += 1
            elif (new_num not in test_data[5]) and (len(test_data[5])<150):
                test_data[5].append(new_num)
                count += 1
            elif (new_num not in test_data[6]) and (len(test_data[6])<150):
                test_data[6].append(new_num)
                count += 1
            elif (new_num not in test_data[7]) and (len(test_data[7])<150):
                test_data[7].append(new_num)
                count += 1
            elif (new_num not in test_data[8]) and (len(test_data[8])<150):
                test_data[8].append(new_num)
                count += 1
            elif (new_num not in test_data[9]) and (len(test_data[9])<150):
                test_data[9].append(new_num)
                count += 1
        for i in range(1522):
            if i not in test_data[0]:
                train_data[0].append(i)
            if i not in test_data[1]:
                train_data[1].append(i)
            if i not in test_data[2]:
                train_data[2].append(i)
            if i not in test_data[3]:
                train_data[3].append(i)
            if i not in test_data[4]:
                train_data[4].append(i)
            if i not in test_data[5]:
                train_data[5].append(i)
            if i not in test_data[6]:
                train_data[6].append(i)
            if i not in test_data[7]:
                train_data[7].append(i)
            if i not in test_data[8]:
                train_data[8].append(i)
            if i not in test_data[9]:
                train_data[9].append(i)
        
        test_data = np.array(test_data)
        train_data = np.array(train_data)
        self.mean = np.mean(data.iloc[:,5])
        self.std = np.std(data.iloc[:,5])
        data.iloc[:,5] = (data.iloc[:,5]-self.mean)/self.std
        
        results = np.zeros((1500,2))
        for i in range(10):
            self.gp = gp_model(data.iloc[train_data[i],[0,1,2,3]], 
                          data.iloc[train_data[i],5], 
                          [0.12274117, 0.08612411, 0.65729583, 0.23342798], 
                          0.16578065, 0.1, 4, 'SE')
            out = self.predict(np.array(data_1.iloc[test_data[i],[0,1,2,3]]))
            results[i*150:(i+1)*150,0] = out
            results[i*150:(i+1)*150,1] = data.iloc[test_data[i],5] * self.std + self.mean
        
        self.setup()
        
        results_all = np.zeros((1522,2))
        
        results_all[:,1] = data.iloc[:,5] * self.std + self.mean
        results_all[:,0] = self.predict(np.array(data_1.iloc[:,[0,1,2,3]]))
        
        return results, results_all

class TC_GP():
    def __init__(self):
        self.y_mean = []
        self.y_std = []
        self.y_max = []
        self.tc_gp = []
        self.setup()
        
    def setup(self):
        data = pd.read_excel("data/tc_data.xlsx")
        x_train = np.array(data.iloc[:,1:5])
        x_train[:,0] = (x_train[:,0]-650)/200
        x_train[:,1] = 100*x_train[:,1]
        x_train[:,2] = 100*x_train[:,2]/2
        x_train[:,3] = 100*x_train[:,3]/3
        
        l_param_list = [[np.sqrt(0.28368), np.sqrt(0.44255), np.sqrt(0.19912), np.sqrt(5.48465)],
                        [np.sqrt(2.86816), np.sqrt(2.57049), np.sqrt(0.64243), np.sqrt(94.43864)],
                        [np.sqrt(6.41552), np.sqrt(12.16391), np.sqrt(7.16226), np.sqrt(27.87327)],
                        [np.sqrt(34.57352), np.sqrt(12.83549), np.sqrt(4.73291), np.sqrt(275.83489)]]
        sf_list = [4*1.57933, 4*5.5972, 4*78.32377, 4*14.79803]
    
        for k in range(4):
            self.y_mean.append(np.mean(np.array(data.iloc[:,k+5])))
            self.y_max.append(np.max(np.array(data.iloc[:,k+5])))
            self.y_std.append(np.std(np.array(data.iloc[:,k+5])))
            y_train = (np.array(data.iloc[:,k+5])-self.y_mean[k])/self.y_std[k]
            l_param = l_param_list[k]
            sf = sf_list[k]
            self.tc_gp.append(gp_model(x_train, y_train, np.array(l_param), sf, 0.05, 4, 'M52'))
            
    def TC_GP_Predict(self, index, x_predict):
        # x_predict = np.expand_dims(x_predict, 0)
        y_out, y_out_var = self.tc_gp[index].predict_var(x_predict)
        
        y_pred = y_out*self.y_std[index] + self.y_mean[index]
        
        y_pred[np.where(y_pred<0)] = 0
        y_pred[np.where(y_pred>self.y_max[index])] = self.y_max[index]
        return y_pred
        
        # if y_pred < 0:
        #     return 0
        # elif y_pred > self.y_max[index]:
        #     return self.y_max[index]
        # else:
        #     return y_pred
    
    def predict(self, x_predict):
        if len(x_predict.shape) == 1:
            x_predict = np.expand_dims(x_predict, axis=0)
        x = np.ones((x_predict.shape[0],4))
        x[:,0] = (x_predict[:,0]-650)/200 #Temperature
        x[:,1] = x_predict[:,1]           #wt% C
        x[:,2] = x[:,2]*0.283/2         #wt% Si
        x[:,3] = x[:,3]*0.328/3         #wt% Mn
        
        vf = self.TC_GP_Predict(0, x)
        xC = self.TC_GP_Predict(1, x)
        xSi = self.TC_GP_Predict(2, x)
        xMn = self.TC_GP_Predict(3, x)
        
        vf_ferr = 1-vf
        xMn_ferr = np.zeros_like(vf_ferr)
        xSi_ferr = np.zeros_like(vf_ferr)
        
        xMn_ferr[np.where(vf_ferr>1e-6)] = (x[np.where(vf_ferr>1e-6),3]/100-vf[np.where(vf_ferr>1e-6)]*xMn[np.where(vf_ferr>1e-6)])/vf_ferr[np.where(vf_ferr>1e-6)]
        xSi_ferr[np.where(vf_ferr>1e-6)] = (x[np.where(vf_ferr>1e-6),2]/100-vf[np.where(vf_ferr>1e-6)]*xSi[np.where(vf_ferr>1e-6)])/vf_ferr[np.where(vf_ferr>1e-6)]
        
        xMn_ferr[np.where(xMn_ferr<0)] = 0
        xSi_ferr[np.where(xSi_ferr<0)] = 0 
        
        xMn_ferr[np.where(xMn_ferr>x[:,3]/100)] = x[np.where(xMn_ferr>x[:,3]/100),3]/100
        xSi_ferr[np.where(xSi_ferr>x[:,2]/100)] = x[np.where(xSi_ferr>x[:,2]/100),2]/100 
        
        return np.array([vf,xC,xMn_ferr,xSi_ferr]).transpose()
                
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


def predict_low_order_model(tc_gp, x_predict, model):
    ep = 0.009
    tc_out = tc_gp.predict(x_predict)
    if model == "isostrain": 
        return isostrain_IS(tc_out, ep)
    if model == "isostress": 
        return isostress_IS(tc_out, ep)
    if model == "isowork": 
        return isowork_IS(tc_out, ep)
    
class tc_vf_classifier():
    def __init__(self):
        self.setup()
        
    def setup(self):
        tc_gp = TC_GP()
    
        size = 200
        
        temp = np.linspace(0,1,size,endpoint=True)
        
        x_fused = np.zeros((size*size,2))
        for i in range(size):
            for j in range(size):
                x_fused[i*size+j,0] = temp[i]
                x_fused[i*size+j,1] = temp[j]
                
        x_fused[:,0] = x_fused[:,0]*200 + 650
        tc_out = tc_gp.predict(x_fused)
        
        output = np.zeros((size*size,))
        output[np.nonzero(tc_out[:,0] > 0.9)] = 1
        
        from sklearn.tree import DecisionTreeClassifier
        self.clf = DecisionTreeClassifier()
        self.clf.fit(x_fused,output)
    
    def predict(self, x_fused):
        return self.clf.predict(x_fused)

def k_medoids(sample, num_clusters):
    D = scipy.spatial.distance_matrix(sample, sample)
    M, C = kMedoids(D, num_clusters)
    return M, C

def calculate(process_name, tasks, results):
    # this multiprocess work will calculate the knowledge gradient choice for
    # a single set of hyper-parameters.
    while True:
        (finish, model_temp, x_fused, fused_model_HP, \
         kernel, x_test, jj, kk, mm, true_sample_count) = tasks.get()
        if finish < 0:
            results.put(-1)
            break
        else:
            cost = [0.246179, 0.890249,  1.827838]
            output = [0,0,0,0,0,jj,kk,mm,0,0]
            model_temp.create_fused_GP(x_fused, fused_model_HP[0:2], 
                                        fused_model_HP[2], 0.1, 
                                        kernel)
            fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
            
            index_max = np.nonzero(fused_mean == np.max(fused_mean))
            output[0] = np.max(fused_mean)
            output[1] = x_test[index_max,0]
            output[2] = x_test[index_max,1]
            
            nu_star, x_star, NU = knowledge_gradient(true_sample_count, 
                                                      0.1, 
                                                      fused_mean, 
                                                      fused_var)
            output[3] = nu_star/cost[jj]
            output[4] = x_star
            output[8] = x_test[x_star,0]*200 + 650
            output[9] = x_test[x_star,1]
            results.put(output)

if __name__ == "__main__":     
    param = sys.argv
    # When testing, particularly in Spyder, the following ensures the correct
    # operation of the code
    if len(param) == 1:
        param = ['', 'M52', '2', '10', '50', '2', '1', '14000', '1000']
    """
    This code is to do batch bayesian optimization of two dimensional problem
    within the DEMS project. The assumption at this point is that we have
    enough information to define the parameters for the low order models, but
    we don't have enough information to define the parameters of the fused
    model parameters. Therefore, the batch optimization will be done only on
    the Fused Model GP. However, this approach is still taking the step that
    while we know the ideal parameters for the low order GPs we have only very
    limited information from the low-order models to start with.
    """

    kernel = param[1]           # define the kernel to be used when defining the GPs
    iter_count = int(param[2])  # define the number of iterations
    sample_count = int(param[3])# define the number of samples to test on (modified by classifier)
    hp_count = int(param[4])    # define the number of hyper-parameter sets
    num_medoids = int(param[5]) # define the number of medoid clusters to use
    rve_iter = int(param[6])    # define the number of iterations between each RVE call
    total_budget = int(param[7])# define the total budget 
    rve_budget = int(param[8])  # define the RVE budget
    with open("current_index.txt",'r') as f:
        curr_index = f.read()
    init_index = int(curr_index)
    today = dt.datetime.today()
    date = "{}_{}_{}_{}_{}".format(today.year,today.month,today.day,today.hour,today.minute)
    
    results_dir_name = 'results_Budget_{}m-{}hp-{}sc-{}ri'.format(num_medoids, hp_count, sample_count, rve_iter)
    try:
        os.mkdir('results')
    except FileExistsError:
        pass
    try:
        os.mkdir('results/{}'.format(date))
    except FileExistsError:
        pass
    
    # define the points for creating the fused GP
    points1 = 27
    temp = np.linspace(0,1,points1,endpoint=True)
    x_fused = np.zeros((points1*points1,2))
    for i in range(points1):
        for j in range(points1):
            x_fused[i*points1+j,0] = temp[i]
            x_fused[i*points1+j,1] = temp[j]
    
    temp = deepcopy(x_fused)
    temp[:,0] = temp[:,0]*200 + 650
    clf = tc_vf_classifier()
    
    clf_out = clf.predict(temp)
        
    x_fused = x_fused[np.nonzero(clf_out==0)[0],:]

    random_init = pd.read_csv("data/init_data.csv",header=None)
    random_init = np.array(random_init)
        
    # define the initial data point, this will be the same for all models.
    initial_data = np.array([[random_init[init_index, 0],random_init[init_index, 1]],
                             [random_init[init_index, 2],random_init[init_index, 3]]])
    
    # define the Thermo-Calc GP
    tc_gp = TC_GP()
    # define the RVE GP - This GP is used in lieu of the actual RVE code,
    # a separate GP will be created for the data extracted from the RVE code
    rve_gp = RVE_GP()
    
    rve_out = rve_gp.predict(initial_data)

    # we need the Thermo-Calc output in order to get the initial data for the 
    # low order models
    y_init = []
    x_init = []
    x_init.append(initial_data)
    y_init.append(predict_low_order_model(tc_gp, initial_data, 'isostrain').flatten())
    x_init.append(initial_data)
    y_init.append(predict_low_order_model(tc_gp, initial_data, 'isostress').flatten())
    x_init.append(initial_data)
    y_init.append(predict_low_order_model(tc_gp, initial_data, 'isowork').flatten())
    
    model_index = {'isostrain':0, 
                    'isostress':1, 
                    'isowork':2}
    model_names = ['isostrain','isostress','isowork']
    # Since this is working under the assumption that we know the hyper-parameters
    # for the low order model GPs, it is necessary to set these
    model_l = [[1.01584704, 0.21626703],
                [3.65895877e+00, 3.43182042e+00],
                [1.13256232, 0.26973212]]   
    model_sf = [2.11334467,1.76954682e+04,3.410617]
    model_sn = [0.05, 0.05, 0.05]
    
    err_l = [[0.1, 0.1],
              [0.1, 0.1],
              [0.1, 0.1]]   
    err_sf = [1,1,1]
    err_sn = [0.05, 0.05, 0.05]
    
    model_means = [9.95204, 30.13247, 10.71565]
    model_std = [6.67520, 3.87400, 6.46956]
    
    
    model_control = model_reification(x_init, y_init, model_l, model_sf, 
                                      model_sn, model_means, 
                                      model_std, err_l, err_sf, err_sn, 
                                      initial_data, 
                                      rve_out, 3, 2, kernel)

    fused_model_HP = lhs(3,hp_count)
    fused_model_HP[:,0] = fused_model_HP[:,0]*20 + 0.01
    fused_model_HP[:,1] = fused_model_HP[:,1]*20 + 0.01
    fused_model_HP[:,2] = fused_model_HP[:,2]*99.9 + 0.1
    
    with open("results/{}/{}_model_record.csv".format(date, results_dir_name), 'w') as f:
        f.write("Isostrain,Isostress,Isowork,RVE,\n")
        f.write("0,0,0,0,\n")
        
    model_record = [0,0,0,0]
    iteration_time = [0]
    medoids_list = []
    all_RVE_x = []
    all_RVE_y = []
    max_RVE = [np.max(rve_out)]
    
    with open("results/{}/{}_iteration_data.csv".format(date, results_dir_name), 'w') as f:
        f.write("Iteration,Model,Temperature,Carbon,Model Out,\n")
        f.write("{},{},{},{},{},\n".format(-1,0,initial_data[0,0],initial_data[0,1],y_init[0][0])) 
        f.write("{},{},{},{},{},\n".format(-1,0,initial_data[1,0],initial_data[1,1],y_init[0][1]))
        f.write("{},{},{},{},{},\n".format(-1,1,initial_data[0,0],initial_data[0,1],y_init[1][0]))
        f.write("{},{},{},{},{},\n".format(-1,1,initial_data[1,0],initial_data[1,1],y_init[1][1]))
        f.write("{},{},{},{},{},\n".format(-1,2,initial_data[0,0],initial_data[0,1],y_init[2][0]))
        f.write("{},{},{},{},{},\n".format(-1,2,initial_data[1,0],initial_data[1,1],y_init[2][1]))
        f.write("{},{},{},{},{},\n".format(-1,3,initial_data[0,0],initial_data[0,1],rve_out[0]))
        f.write("{},{},{},{},{},\n".format(-1,3,initial_data[1,0],initial_data[1,1],rve_out[1]))
    
    with open("results/{}/{}_iteration_cost.csv".format(date, results_dir_name), 'w') as f:
        f.write("Model Cost, Total Budget Left, RVE Budget Left,\n")
        
    with open("results/{}/{}_log.txt".format(date, results_dir_name), 'w') as f:
        f.write("Iterations Completed,\n")
         
    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()
    
    # Create process pool with four processes
    num_processes = multiprocessing.cpu_count()
    if num_processes > 25:
        num_processes = 20
    pool = multiprocessing.Pool(processes=num_processes)
    
    rve_Budget_Left = rve_budget
    total_Budget_Left = total_budget
    
    ii = 0
    
    while True:
        with open("results/{}/{}_log.txt".format(date, results_dir_name), 'a') as f:
            f.write("{},\n".format(ii))
        start_iteration = time()

        start = time()
        x_test = lhs(2, sample_count)
        x_test1 = deepcopy(x_test)
        x_test1[:,0] = x_test1[:,0]*200 + 650
        
        clf_out = clf.predict(x_test1)
        
        x_test = x_test[np.nonzero(clf_out==0)[0],:]
        x_test1 = x_test1[np.nonzero(clf_out==0)[0],:]
        
        true_sample_count = x_test1.shape[0]
        
        tc_out = tc_gp.predict(x_test1)
               
        x_test.shape
        new_mean = []
        
        # obtain predictions for the mechanical properties from the low-order
        # GPs
        new, var = model_control.predict_low_order(x_test, 
                                                    model_index['isostrain'])
        new_mean.append(new)
        new, var = model_control.predict_low_order(x_test, 
                                                    model_index['isostress'])
        new_mean.append(new)
        new, var = model_control.predict_low_order(x_test, 
                                                    model_index['isowork'])
        new_mean.append(new)
                
        kg_output = [] 
        
        processes = []
        # Initiate the worker processes
        for i in range(num_processes):
        
            # Set process name
            process_name = 'P%i' % i
        
            # Create the process, and connect it to the worker function
            new_process = multiprocessing.Process(target=calculate, args=(process_name,tasks,results))
        
            # Add new process to the list of processes
            processes.append(new_process)
        
            # Start the process
            new_process.start()
        
        # Calculate the Knowledge Gradient for each of the test points in each
        # model for each set of hyperparameters
        for jj in range(3):
            for kk in range(true_sample_count):
                model_temp = deepcopy(model_control)
                model_temp.update_GP(np.expand_dims(x_test[kk], axis=0), 
                                      np.expand_dims(np.array([new_mean[jj][kk]]), 
                                                axis=0), jj)
    
                # model_temp.update_GP(x_test[kk], 
                #                      np.array([new_mean[jj][kk]]), 0)
                for mm in range(hp_count):
                    single_task = (1, model_temp, x_fused, fused_model_HP[mm,:],
                                    kernel, x_test, jj, kk, mm, true_sample_count)
                    tasks.put(single_task)
                    
        # Wait while the workers process
        sleep(60)
        
        # Quit the worker processes by sending them -1
        for i in range(num_processes):
            tasks.put((-1,1,1,1,1,1,1,1,1,1))
            
        # Read calculation results
        num_finished_processes = 0
        while True:
            # Read result
            new_result = results.get()
        
            # Have a look at the results
            if new_result == -1:
                # Process has finished
                num_finished_processes += 1
        
                if num_finished_processes == num_processes:
                    break
            else:
                # Output result
                kg_output.append(new_result)
                
        for i in range(num_processes):
            processes[i].terminate()

        # convert to a numpy array for ease of indexing
        kg_output = np.array(kg_output)

        point_selection = {}
        selected_indices = []
        for iii in range(kg_output.shape[0]):
            try:
                if kg_output[iii,5] in point_selection[kg_output[iii,4]]['models']:
                    if kg_output[iii,3] > point_selection[kg_output[iii,4]]['nu'][kg_output[iii,5]]:
                        point_selection[kg_output[iii,4]]['nu'][kg_output[iii,5]] = kg_output[iii,3]
                        point_selection[kg_output[iii,4]]['kg_out'][kg_output[iii,5]] = iii
                else:
                    point_selection[kg_output[iii,4]]['models'].append(kg_output[iii,5])
                    point_selection[kg_output[iii,4]]['nu'][kg_output[iii,5]] = kg_output[iii,3]
                    point_selection[kg_output[iii,4]]['kg_out'][kg_output[iii,5]] = iii
            except KeyError:
                point_selection[kg_output[iii,4]] = {'models':[kg_output[iii,5]],
                                                     'nu':[-1e6,-1e6,-1e6],
                                                     'kg_out':[-1,-1,-1]}
                point_selection[kg_output[iii,4]]['nu'][kg_output[iii,5]] = kg_output[iii,3]
                point_selection[kg_output[iii,4]]['kg_out'][kg_output[iii,5]] = iii
        
        med_input = [[],[],[],[]]        
        for index in point_selection.keys():
            for jjj in range(len(point_selection[index]['models'])):
                med_input[0].append(point_selection[index]['nu'][point_selection[index]['models'][jjj]])
                med_input[1].append(index)
                med_input[2].append(point_selection[index]['models'][jjj])
                med_input[3].append(point_selection[index]['kg_out'][point_selection[index]['models'][jjj]])
        med_input = np.array(med_input).transpose()
                   
        # Since there may be too many duplicates when using small numbers of
        # test points and hyper-parameters check to make sure and then return
        # all the points if there are less than the required number of points
        if med_input.shape[0] > num_medoids:
            medoids, clusters = k_medoids(med_input[:,0:3], num_medoids)
        else:
            medoids, clusters = k_medoids(med_input[:,0:3], int(med_input.shape[0]/3))       
        
        # next, need to get the true values for each of the medoids and update the
        # models before starting next iteration.

        medoid_index = []
        for i in range(len(medoids)):
            medoid_index.append(int(med_input[medoids[i],3]))
        medoid_out = kg_output[medoid_index,:]
                
        model_iter_calls = [0,0,0,0]
        
        # When RVE Budget has been exhausted, update the RVE model with 
        # all of the medoid positions, for every other iteration, 
        # update the individual low order models
        
        model_cost = time() - start_iteration
        
        rve_Budget_Left -= model_cost
        total_Budget_Left -= model_cost
        
        if rve_Budget_Left < 0:
            max_new = 0
            for iii in range(len(medoids)):
                x_new = np.array(medoid_out[iii,[8,9]])
                x_new = np.expand_dims(x_new, 0)
                y_new = rve_gp.predict(x_new)
                all_RVE_x.append(x_new)
                all_RVE_y.append(y_new)
                if y_new > max_new:
                    max_new = y_new
                model_control.update_truth(x_new, y_new)
                model_iter_calls[3] += 1
                with open("results/{}/{}_iteration_data.csv".format(date, results_dir_name), 'a') as f:
                    f.write("{},{},{},{},{},\n".format(ii,3,medoid_out[iii,8],medoid_out[iii,9],y_new))
                
                total_Budget_Left -= 7200
                rve_Budget_Left = rve_budget
            sample_count += 5
            if max_new > max_RVE[ii]:
                max_RVE.append(max_new)
            else: 
                max_RVE.append(max_RVE[ii])
        else:
            max_RVE.append(max_RVE[ii])
            # Obtain the results from the medoids for the lower order models
            cost = [0.246179, 0.890249,  1.827838]
            for iii in range(len(medoids)):
                x_new = np.array(medoid_out[iii,[8,9]])
                x_new = np.expand_dims(x_new, 0)
                y_new = predict_low_order_model(tc_gp, x_new, 
                                                model_names[medoid_out[iii,5]])[0,0]
                model_control.update_GP(x_new, y_new, medoid_out[iii,5])
                model_iter_calls[medoid_out[iii,5]] += 1
                with open("results/{}/{}_iteration_data.csv".format(date, results_dir_name), 'a') as f:
                    f.write("{},{},{},{},{},\n".format(ii,medoid_out[iii,5],medoid_out[iii,8],medoid_out[iii,9],y_new))
                total_Budget_Left -= cost[medoid_out[iii,5]]
                rve_Budget_Left -= cost[medoid_out[iii,5]]
        
        for jjj in range(4):
            model_record[jjj] += model_iter_calls[jjj]
            
        with open("results/{}/{}_model_record.csv".format(date, results_dir_name), 'a') as f:
            f.write("{},{},{},{},\n".format(model_record[0],model_record[1],model_record[2],model_record[3]))
            
        with open("results/{}/{}_iteration_cost.csv".format(date, results_dir_name), 'a') as f:
            f.write("{},{},{},\n".format(model_cost, total_Budget_Left, rve_Budget_Left))

        iteration_time.append(time()-start)

        if (total_Budget_Left < 0) or (ii > iter_count):
            break
        
        ii += 1
    
    print("** Code Finished **")
    with open("results/{}/{}_code_finished.txt".format(date, results_dir_name), 'w') as f:
        f.write("** Code Finished **\n")
                
        
        
        
        
        