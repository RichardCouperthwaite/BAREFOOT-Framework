# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:51:59 2021

@author: Richard Couperthwaite
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares
import matplotlib.pyplot as plt
from pickle import load
from gpModel import gp_model
from copy import deepcopy

def plotResults(calcName):
    standData = pd.read_csv("data/StandardTestData.csv")
    
        
    # plt.figure()
    # plt.fill_between(cum_times[6:], np.abs(avg_max[6:])+2*std_max[6:], np.abs(avg_max[6:])-2*std_max[6:], alpha=0.5)
    # plt.plot(cum_times[6:], np.abs(avg_max[6:]))
    # plt.ylim([0,10])
    
    
    pass


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


class RVE_GP():
    def __init__(self):
        self.mean = 0
        self.std = 0
        self.gp = 0
        self.setup()
        
    def setup(self):
        data = pd.read_excel('./data/rve_data.xlsx')
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
        # x = np.ones((x_predict.shape[0],4))        
        # x[:,0] = (x_predict[:,0]-650)/200 #Temperature
        # x[:,1] = x_predict[:,1]           #wt% C
        # x[:,2] = x[:,2]/2           #wt% Si
        # x[:,3] = x[:,3]/3           #wt% Mn
        mean, var = self.gp.predict_var(x_predict)
        
        return (mean*self.std + self.mean)
    
    def test_fit(self):
        data = pd.read_excel('../data/rve_data.xlsx')
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
        data = pd.read_excel("./data/tc_data.xlsx")
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
        x[:,2] = x[:,2]/2         #wt% Si
        x[:,3] = x[:,3]/3         #wt% Mn
        
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


def isostrain_IS(x,ep):
    beta_Si = 732.7676
    beta_Mn = 213.4494
    # beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    s0F = np.zeros((mm,1))
    s0M = np.zeros((mm,1))
    sF = np.zeros((mm,1))
    sM = np.zeros((mm,1))
    stress = np.zeros((mm,10001))
    str_ = np.zeros((mm,1))
    dsde = np.zeros((mm,1))
    cc = np.zeros((mm,))
    index = np.zeros((mm,90))
    
    for ii in range(mm):
        # yield strength of the phases
        s0F[ii]=200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        s0M[ii]=400+1000*((100*x_C[ii])**(1/3))
       
        kF=2200
        kM=450
        nF=0.5
        nM=0.06
        strain=np.linspace(0,1,10001,endpoint=True)
        for i in range(10001):
            sF[ii]=s0F[ii]+kF*strain[i]**nF
            sM[ii]=s0M[ii]+kM*strain[i]**nM
            stress[ii,i]=((1-f[ii])*sF[ii])+(f[ii]*sM[ii])
 
        index[ii,:] = np.array(np.nonzero(strain <= ep))
        str_[ii]=stress[ii,int(np.max(index[ii,:]))]
        dsde[ii]=(stress[ii,int(np.max(index[ii,:]))+1]-stress[ii,int(np.max(index[ii,:]))-1])/(2*(strain[int(np.max(index[ii,:]))+1]-strain[int(np.max(index[ii,:]))]))
        cc[ii]=dsde[ii]/str_[ii]
    
    if single_calc:
        return cc[0]
    else:
        return cc

def isostress_IS(x,ep):
    beta_Si = 732.7676
    beta_Mn = 213.4494
    # beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    s0F = np.zeros((mm,1))
    s0M = np.zeros((mm,1))
    # sF = np.zeros((mm,1))
    # sM = np.zeros((mm,1))
    # stress = np.zeros((mm,10001))
    str_ = np.zeros((mm,1))
    dsde = np.zeros((mm,1))
    cc = np.zeros((mm,))
        
    for ii in range(mm):
        # yield strength of the phases
        s0F[ii]=200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        s0M[ii]=400+1000*((100*x_C[ii])**(1/3))
        vf=f[ii]

        kF=2200
        kM=450
        nF=0.5
        nM=0.06
    
        # Overall Stress
        stress=np.linspace(170,1900,173000,endpoint=True)
        l=len(stress)
        strain = np.zeros((l,1))
        for i in range(l):
            if (stress[i] < s0F[ii]):
                epF=0;
            else:
                epF=((stress[i]-s0F[ii])/kF)**(1/nF)

            if (stress[i] < s0M[ii]):
                epM=0
            else:
                epM=((stress[i]-s0M[ii])/kM)**(1/nM);

            strain[i]=((1-vf)*epF)+(vf*epM);
        
        index = np.array(np.nonzero(strain <= ep))
        str_=stress[np.max(index)];
        dsde=(stress[np.max(index)+1]-stress[np.max(index)-1])/(2*(strain[np.max(index)+1]-strain[np.max(index)]))
        
        cc[ii]=dsde/str_
    if single_calc:
        return cc[0]
    else:
        return cc

def isowork_IS(x,ep):
    beta_Si = 732.7676
    beta_Mn = 213.4494
    # beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    cc = np.zeros((mm,))
        
    for ii in range(mm):
        # yield strength of the phases
        s0F=200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        s0M=400+1000*((100*x_C[ii])**(1/3))
        vf=f[ii]
        kF=2200
        kM=450
        nF=0.5
        nM=0.06
        # strain increment in ferrite
        depF=0.0001
        epF=np.zeros((10000,1))
        epM=np.zeros((10000,1))
        sF=np.ones((10000,1))*s0F
        sM=np.ones((10000,1))*s0M
        sT=np.zeros((10000,1))
        epT=np.zeros((10000,1))
        SS = np.zeros((10000,2))
        for k in range(9999): #i=2:(1/depF)
            i = k+1
            epF[i]=epF[i-1]+depF
            sF[i]=s0F+kF*epF[i]**nF
            wF=sF[i]*depF
            temp=epM[i-1]
            isow =  lambda wF,s0M,kM,nM,temp,depM : wF-((s0M+kM*(temp+depM)**nM)*depM)
            # isow=@(wF,s0M,kM,nM,temp,depM) wF-((s0M+kM*(temp+depM)^nM)*depM)
            fun = lambda depM : isow(wF,s0M,kM,nM,temp,depM)
            # fun=@(depM) isow(wF,s0M,kM,nM,temp,depM)
            depM=fsolve(fun,depF) # depF is initial guess
            epM[i]=epM[i-1]+depM
            sM[i]=s0M+kM*epM[i]**nM
            sT[i]=((1-vf)*sF[i])+(vf*sM[i])
            epT[i]=((1-vf)*epF[i])+(vf*epM[i])
            SS[i,0]=epT[i]
            SS[i,1]=sT[i]
        
        strain=np.zeros((10000,1))
        stress=np.zeros((10000,1))
        
        for iii in range(10000):
            strain[iii]=SS[iii,0]
            stress[iii]=SS[iii,1]
        
        index = np.array(np.nonzero(strain <= ep))
        str_=stress[np.max(index)];
        dsde=(stress[np.max(index)+1]-stress[np.max(index)-1])/(2*(strain[np.max(index)+1]-strain[np.max(index)]))
        
        cc[ii]=dsde/str_
        
    if single_calc:
        return cc[0]
    else:
        return cc

def EC_Mart_IS(x,ep):

    # 0 represents - Matrix
    # 1 represents - inclusions
    
    # Input Variables
    beta_Si = 732.7676
    beta_Mn = 213.4494
    # beta_C = 7507.582;
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    
    cc = np.zeros((mm))
    
    for ii in range(mm):
        
        vf=f[ii]
        
        # Ferrite
        E_1 = 200*10**3
        PR_1 = 0.3
        Mu_1 = E_1/(2*(1+PR_1))
        sigy_1 = 200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        h_1=2200
        n_1=0.5
        
        # Martensite (Matrix) Matrix yields first
        E_0 = 200*10**3
        PR_0 = 0.3
        Mu_0 = E_0/(2*(1+PR_0))
        sigy_0 = 400+1000*((100*x_C[ii])**(1/3))
        h_0=450
        n_0=0.06
        
        # Composition of Phases
        c_0 = vf 
        c_1 = 1-c_0
        
        # Alpha and Beta Values
        # Ferrite
        # alpha_0 = (1/3)*((1+PR_0)/(1-PR_0))
        beta_0 = (2/15)*((4-5*PR_0)/(1-PR_0))
        
        # Austenite
        # alpha_1 = (1/3)*((1+PR_1)/(1-PR_1))
        # beta_1 = (2/15)*((4-5*PR_1)/(1-PR_1))
        
        #Plastic Strain in Matrix
        strain_p_1 = np.linspace(0, 0.2, num=2000, endpoint=True)
        
        # Elastic stage
        
        Mu_0 = E_0/(2*(1+PR_0))
        Mu_1 = E_1/(2*(1+PR_1))
        
        # K_0 = E_0/(3*(1-2*PR_0))
        
        # K_1 = E_1/(3*(1-2*PR_1))
        
        # K = K_0*(1 + (c_1*(K_1-K_0))/(c_0*alpha_0*(K_1-K_0) + K_0))
        
        # Mu = Mu_0*(1 + (c_1*(Mu_1-Mu_0))/(c_0*beta_0*(Mu_1-Mu_0) + Mu_0))
    
        # E = 9*K*Mu/(3*K+Mu)
        
        # a_0 = (alpha_0*(K_1-K_0) + K_0)/((c_1 + (1-c_1)*alpha_0)*(K_1-K_0)+ K_0)
        b_0 = (beta_0*(Mu_1-Mu_0) + Mu_0)/((c_1 + (1-c_1)*beta_0)*(Mu_1-Mu_0)+ Mu_0)
        # a_1 =  K_1/((c_1 + (1-c_1)*alpha_0)*(K_1-K_0)+ K_0)
        b_1 = Mu_1/((c_1 + (1-c_1)*beta_0)*(Mu_1-Mu_0)+ Mu_0)
        
        strain_p_0 = np.zeros((len(strain_p_1)))
        count=0
        SS=np.zeros((len(strain_p_1),2))
        
        strain_c = np.zeros((len(strain_p_1)))
        stress_c = np.zeros((len(strain_p_1)))
        
        for i in range(len(strain_p_1)):
                        
            strain_c[i] = c_1*b_1*strain_p_1[i]
            
            stress_c[i] = (1/b_1)*(sigy_1 + h_1*(strain_p_1[i]**n_1) + 3*Mu_0*(1-beta_0)*(c_0*b_1*strain_p_1[i]))
            
            temp = (1/b_0)*(sigy_0 - 3*Mu_0*(1-beta_0)*strain_c[i])
            
            if (stress_c[i]>(temp+150)) or (c_1 == 0):
                
                count=count+1
                
                A = b_0
                B = 3*Mu_0*(1-beta_0)*c_1*b_1
                C = sigy_1 + h_1*(strain_p_1[i]**n_1)
                D = b_1
                G = 3*Mu_0*(1-beta_0)*(1-c_1)*b_1
                
                x0=np.random.rand(2)                
                

                
                F = lambda y : [-A*y[0] + B*y[1] + sigy_0 + h_0*(y[1]**n_0) - B*strain_p_1[i], -D*y[0] - G*y[1] + C + G*strain_p_1[i]]
                

                
                
                y = least_squares(F, x0, bounds=((0,0),(np.inf,np.inf))).x

                stress_c[i] = y[0];
                strain_p_0[i]= y[1];
                
                strain_c[i] = c_0*b_0*strain_p_0[i] + c_1*b_1*strain_p_1[i]
                
                

        
        SS[:,0] = strain_c
        SS[:,1] = stress_c
    
        strain=np.zeros((len(strain_p_1)))
        stress=np.zeros((len(strain_p_1)))
        
        for iii in range(SS.shape[0]):
            strain[iii] = SS[iii,0]
            stress[iii] = SS[iii,1]
        
        index = np.where(strain <= ep)
        
        strs=stress[np.max(index)]
        dsde=(stress[np.max(index)+1]-stress[np.max(index)-1])/(2*(strain[np.max(index)+1]-strain[np.max(index)]))

        cc[ii]=dsde/strs
        
    if single_calc:
        return cc[0]
    else:
        return cc

def secant1_IS(x,ep):

    # Input Variables
    beta_Si = 732.7676
    beta_Mn = 213.4494
    # beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    cc = np.zeros((mm))
    
    for ii in range(mm):
        
        vf = f[ii]
        
        # # Ferrite (Matrix)
        E_a = 200*10**3

        PR_a = 0.3
        Mu_a = E_a/(2*(1+PR_a))
        sigy_a = 200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        h_a=2200
        n_a=0.5
        
        # Martensite
        E_f = 200*10**3
        PR_f = 0.3
        Mu_f = E_f/(2*(1+PR_f))
        sigy_f = 400+1000*((100*x_C[ii])**(1/3))
        h_f=450
        n_f=0.06
        
        # Composition of Phases
        c_f = vf
        c_a = 1-c_f
        
        # Alpha and Beta Values
        # Austenite
        alpha_a = (1/3)*((1+PR_a)/(1-PR_a))
        beta_a = (2/15)*((4-5*PR_a)/(1-PR_a))
        
        #Ferrite
        # alpha_f = (1/3)*((1+PR_f)/(1-PR_f))
        # beta_f = (2/15)*((4-5*PR_f)/(1-PR_f))
        
        
        #Plastic Strain in Matrix
        strain_p_a = np.linspace(0,0.17,num=340,endpoint=True)
        
        
        # Elastic stage
        K_a = E_a/(3*(1-2*PR_a));
        
        K_f = E_f/(3*(1-2*PR_f));
        
        K = K_a*(1 + (c_f*(K_f-K_a))/(c_a*alpha_a*(K_f-K_a) + K_a));
        
        Mu = Mu_a*(1 + (c_f*(Mu_f-Mu_a))/(c_a*beta_a*(Mu_f-Mu_a) + Mu_a));
        
        E = 9*K*Mu/(3*K+Mu);
        
        # a_a = (alpha_a*(K_f-K_a) + K_a)/((c_f + (1-c_f)*alpha_a)*(K_f-K_a)+ K_a);
        # b_a = (beta_a*(Mu_f-Mu_a) + Mu_a)/((c_f + (1-c_f)*beta_a)*(Mu_f-Mu_a)+ Mu_a);
        # a_f =  K_f/((c_f + (1-c_f)*alpha_a)*(K_f-K_a)+ K_a);
        # b_f = Mu_f/((c_f + (1-c_f)*beta_a)*(Mu_f-Mu_a)+ Mu_a);
        
        count1=0
        
        # Starting with a given plastic strain in the matrix and then
        # increasing the value
        strain_p_f=np.zeros(strain_p_a.shape[0])
        strain_p_c=np.zeros(strain_p_a.shape[0])
        E_s_a=np.zeros(strain_p_a.shape[0])
        PR_s_a=np.zeros(strain_p_a.shape[0])
        Mu_s_a=np.zeros(strain_p_a.shape[0])
        alpha_s_a=np.zeros(strain_p_a.shape[0])
        beta_s_a=np.zeros(strain_p_a.shape[0])
        b_s_a=np.zeros(strain_p_a.shape[0])
        b_s_f=np.zeros(strain_p_a.shape[0])
        K_s=np.zeros(strain_p_a.shape[0])
        Mu_s=np.zeros(strain_p_a.shape[0])
        E_s=np.zeros(strain_p_a.shape[0])
        stress_c=np.zeros(strain_p_a.shape[0])
        SS=np.zeros((strain_p_a.shape[0],2))
        
        
        for j in range(strain_p_a.shape[0]):
            
            count1 = count1+1;
            
            # Secant Modulus given by Eq 2.8
            E_s_a[j] = 1/((1/E_a) + strain_p_a[j]/(sigy_a + h_a*(strain_p_a[j])**n_a))
            PR_s_a[j] = 0.5 - ((0.5 - PR_a)*(E_s_a[j]/E_a))
            Mu_s_a[j] = E_s_a[j]/(2*(1+PR_s_a[j]))
            
            # Austenite
            alpha_s_a[j] = (1/3)*((1+PR_s_a[j])/(1-PR_s_a[j]))
            beta_s_a[j] = (2/15)*((4-5*PR_s_a[j])/(1-PR_s_a[j]))
            
            b_s_a[j] = (beta_s_a[j]*(Mu_f-Mu_s_a[j]) + Mu_s_a[j])/((c_f + (1-c_f)*beta_s_a[j])*(Mu_f-Mu_s_a[j])+ Mu_s_a[j])
            b_s_f[j] = Mu_f/((c_f + (1-c_f)*beta_s_a[j])*(Mu_f-Mu_s_a[j])+ Mu_s_a[j])
            
            
            K_s[j] = K_a*(1+ ((c_f*(K_f-K_a))/((1-c_f)*alpha_s_a[j]*(K_f-K_a) + K_a)))
            Mu_s[j] = Mu_s_a[j]*(1+ ((c_f*(Mu_f-Mu_s_a[j]))/((1-c_f)*beta_s_a[j]*(Mu_f-Mu_s_a[j]) + Mu_s_a[j])))
            
            E_s[j] = (9*K_s[j]*Mu_s[j])/(3*K_s[j] + Mu_s[j])
            
            # Total stress and plastic strain of composite
            stress_c[j] = ((1/b_s_a[j])*(sigy_a + h_a*((strain_p_a[j])**n_a)))
            
            
            if (stress_c[j]-(sigy_f/b_s_f[j])) > 110:
                A = b_s_a[j]
                B = 3*Mu_s_a[j]*(1-beta_s_a[j])*c_f*b_s_f[j]
                C = sigy_a + h_a*(strain_p_a[j]**n_a)
                D = b_s_f[j]
                G = 3*Mu_s_a[j]*(1-beta_s_a[j])*(1-c_f)*b_s_f[j]
                
                x0=np.random.rand(2)              
                
                F = lambda x : [A*x[0] + B*x[1] - C, D*x[0] - G*x[1] - sigy_f - h_f*((x[1])**n_f)];
            
                x = least_squares(F,x0,bounds=((0,0),(np.inf,np.inf)),max_nfev=200000,ftol=1e-60,xtol=3e-9).x

                stress_c[j] = x[0]
                strain_p_f[j]= x[1]
                #
                strain_p_c[j] = c_f*b_s_f[j]*x[1] + (2/3)*(1/(2*Mu_s[j]) - 1/(2*Mu))*stress_c[j]
            else:
                strain_p_c[j] = ((1/E_s[j]) - (1/E))*stress_c[j]
        
            SS[j,1] = stress_c[j]
            SS[j,0] = strain_p_c[j]
            
        
        
        strain=np.zeros((len(SS)))
        stress=np.zeros((len(SS)))
        
        for iii in range(len(SS)):
            strain[iii]=SS[iii,0]
            stress[iii]=SS[iii,1]
        

        
        index = np.where(strain <= ep);
        strs=stress[np.max(index)]
        dsde=(stress[np.max(index)+1]-stress[np.max(index)-1])/(2*(strain[np.max(index)+1]-strain[np.max(index)]));
        

        cc[ii]=dsde/strs
    
    if single_calc:
        return cc[0]
    else:
        return cc