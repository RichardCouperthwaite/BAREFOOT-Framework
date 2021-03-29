# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:18:52 2021

@author: Richard Couperthwaite
"""

from barefoot import barefoot
import sys
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares
import matplotlib.pyplot as plt
from pickle import load, dump
from testFrameworkCodeUtil import ThreeHumpCamel, ThreeHumpCamel_LO1, ThreeHumpCamel_LO2, ThreeHumpCamel_LO3, plotResults 
from testFrameworkCodeUtil import isostress_IS, isostrain_IS, isowork_IS, EC_Mart_IS, secant1_IS, TC_GP, RVE_GP

def thcTest():
    """
    This test sample is run on the TAMU HPRC Ada Cluster with 1 node and 20 cores
    Total requested memory for the job: 51200.00 MB
    """

    
    acquisitionFunc = ["EI", "KG", "TS"]
    for jj in range(3):
        for ii in range(20):
            ROMList = [ThreeHumpCamel_LO1, ThreeHumpCamel_LO2, ThreeHumpCamel_LO3]
            
            calcName = "frameworkTest-{}{}".format(acquisitionFunc[jj], ii)
            dataPath = "data/testData/initData{}".format(ii)
            
            framework = barefoot(ROMModelList=ROMList, TruthModel=ThreeHumpCamel, 
                            calcInitData=False, initDataPathorNum=dataPath, nDim=2, 
                            calculationName=calcName, acquisitionFunc=acquisitionFunc[jj],
                            restore_calc=False, logname="frameworkTest{}{}".format(acquisitionFunc[jj], jj*20 + ii), multiNode=5)
            
            modelParam = {'model_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                          'model_sf': [1,1,1,],
                          'model_sn': [0.01, 0.01, 0.01],
                          'means': [0,0,0],
                          'std': [1,1,1],
                          'err_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                          'err_sf': [1,1,1],
                          'err_sn': [0.1, 0.1, 0.1],
                          'costs': [0.9, 1.1, 5, 5000]}
            
            framework.initialize_parameters(modelParam=modelParam, covFunc="M52", iterLimit=1,  
                                              sampleCount=20, hpCount=30, batchSize=5, 
                                              tmIter=1, totalBudget=1e16, tmBudget=1e16, 
                                              upperBound=1, lowBound=0.0001, fusedPoints=10)
            
            framework.run_optimization()
        
def call_isowork(x):
    ep = 0.009
    x = np.expand_dims(x, axis=0)
    x[:,0] = 200*x[:,0] + 650
    x[:,2] = 2*x[:,0]
    x[:,3] = 3*x[:,0]
    with open("data/tc_gpObj", 'rb') as f:
        tc_gp = load(f)
    tc_out = tc_gp.predict(x)
    return isowork_IS(tc_out, ep)

def call_isostrain(x):
    ep = 0.009
    x = np.expand_dims(x, axis=0)
    x[:,0] = 200*x[:,0] + 650
    x[:,2] = 2*x[:,0]
    x[:,3] = 3*x[:,0]
    with open("data/tc_gpObj", 'rb') as f:
        tc_gp = load(f)
    tc_out = tc_gp.predict(x)
    return isostrain_IS(tc_out, ep)

def call_isostress(x):
    ep = 0.009
    x = np.expand_dims(x, axis=0)
    x[:,0] = 200*x[:,0] + 650
    x[:,2] = 2*x[:,0]
    x[:,3] = 3*x[:,0]
    with open("data/tc_gpObj", 'rb') as f:
        tc_gp = load(f)
    tc_out = tc_gp.predict(x)
    return isostress_IS(tc_out, ep)

def call_ecMart(x):
    ep = 0.009
    x = np.expand_dims(x, axis=0)
    x[:,0] = 200*x[:,0] + 650
    x[:,2] = 2*x[:,0]
    x[:,3] = 3*x[:,0]
    with open("data/tc_gpObj", 'rb') as f:
        tc_gp = load(f)
    tc_out = tc_gp.predict(x)
    return EC_Mart_IS(tc_out, ep)

def call_secant1(x):
    ep = 0.009
    x = np.expand_dims(x, axis=0)
    x[:,0] = 200*x[:,0] + 650
    x[:,2] = 2*x[:,0]
    x[:,3] = 3*x[:,0]
    with open("data/tc_gpObj", 'rb') as f:
        tc_gp = load(f)
    tc_out = tc_gp.predict(x)
    return secant1_IS(tc_out, ep)
        
        
def mechModelTest():
    rve_gp = RVE_GP() 
          
    acquisitionFunc = ["EI", "KG", "TS"]
    for jj in range(3):
        for ii in range(20):
            ROMList = [call_isowork, call_isostrain, call_isostress, call_ecMart, call_secant1]
            
            calcName = "mechModelTest-{}{}".format(acquisitionFunc[jj], ii)
            
            framework = barefoot(ROMModelList=ROMList, TruthModel=rve_gp.predict, 
                            calcInitData=True, initDataPathorNum=[2,2,2,2,2,2], nDim=4, 
                            calculationName=calcName, acquisitionFunc=acquisitionFunc[jj],
                            restore_calc=False, logname="mechtest{}".format(jj*20 + ii), multiNode=5)
            
            modelParam = {'model_l': [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],
                                      [0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]],
                          'model_sf': [1,1,1,1,1],
                          'model_sn': [0.01, 0.01, 0.01, 0.01, 0.01],
                          'means': [0,0,0,0,0],
                          'std': [1,1,1,1,1],
                          'err_l': [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],
                                      [0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]],
                          'err_sf': [1,1,1,1,1],
                          'err_sn': [0.01, 0.01, 0.01, 0.01, 0.01],
                          'costs': [1,2,3,4,5, 5000]}
            
            framework.initialize_parameters(modelParam=modelParam, covFunc="M52", iterLimit=30,  
                                              sampleCount=20, hpCount=20, batchSize=5, 
                                              tmIter=5, totalBudget=1e16, tmBudget=1e16, 
                                              upperBound=1, lowBound=0.0001, fusedPoints=10)
            
            framework.run_optimization()


def thcTest2(acqFunc, logname, testNum, initNum, tmacqFunc):
    """
    This test sample is run on the TAMU HPRC Ada Cluster with 1 node and 20 cores
    Total requested memory for the job: 51200.00 MB
    """

    ROMList = [ThreeHumpCamel_LO1, ThreeHumpCamel_LO2, ThreeHumpCamel_LO3]
    
    calcName = "frameworkTest-{}{}".format(acqFunc, testNum)
    dataPath = "data/testData/initData{}".format(int(initNum))
    
    framework = barefoot(ROMModelList=ROMList, TruthModel=ThreeHumpCamel, 
                    calcInitData=False, initDataPathorNum=dataPath, nDim=2, 
                    calculationName=calcName, acquisitionFunc=acqFunc,
                    restore_calc=False, logname=logname, tmSampleOpt=tmacqFunc)
    
    modelParam = {'model_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                  'model_sf': [1,1,1,],
                  'model_sn': [0.01, 0.01, 0.01],
                  'means': [0,0,0],
                  'std': [1,1,1],
                  'err_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                  'err_sf': [1,1,1],
                  'err_sn': [0.1, 0.1, 0.1],
                  'costs': [0.9, 1.1, 5, 5000]}
    
    framework.initialize_parameters(modelParam=modelParam, covFunc="M52", iterLimit=30,  
                                      sampleCount=30, hpCount=100, batchSize=5, 
                                      tmIter=5, totalBudget=1e16, tmBudget=1e16, 
                                      upperBound=1, lowBound=0.0001, fusedPoints=10)
    
    framework.run_optimization()
    
def mechModelTest2(acqFunc, logname, testNum, initNum, tmacqFunc):
    rve_gp = RVE_GP() 
          
    ROMList = [call_isowork, call_isostrain, call_isostress]#, call_ecMart, call_secant1]
    
    calcName = "frameworkTest-{}{}".format(acqFunc, testNum)
    dataPath = "data/mechTestData/initMechData{}".format(int(initNum))
    
    framework = barefoot(ROMModelList=ROMList, TruthModel=rve_gp.predict, 
                    calcInitData=False, initDataPathorNum=dataPath, nDim=4, 
                    calculationName=calcName, acquisitionFunc=acqFunc,
                    restore_calc=False, logname=logname, tmSampleOpt=tmacqFunc)
    
    modelParam = {'model_l': [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]],
                  'model_sf': [1,1,1],
                  'model_sn': [0.01, 0.01, 0.01],
                  'means': [0,0,0],
                  'std': [1,1,1],
                  'err_l': [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]],
                  'err_sf': [1,1,1],
                  'err_sn': [0.01, 0.01, 0.01],
                  'costs': [1,2,3, 5000]}
    
    # modelParam = {'model_l': [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],
    #                           [0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]],
    #               'model_sf': [1,1,1,1,1],
    #               'model_sn': [0.01, 0.01, 0.01, 0.01, 0.01],
    #               'means': [0,0,0,0,0],
    #               'std': [1,1,1,1,1],
    #               'err_l': [[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],
    #                           [0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]],
    #               'err_sf': [1,1,1,1,1],
    #               'err_sn': [0.01, 0.01, 0.01, 0.01, 0.01],
    #               'costs': [1,2,3,4,5, 5000]}
    
    framework.initialize_parameters(modelParam=modelParam, covFunc="M52", iterLimit=30,  
                                      sampleCount=20, hpCount=50, batchSize=3, 
                                      tmIter=5, totalBudget=1e16, tmBudget=1e16, 
                                      upperBound=1, lowBound=0.0001, fusedPoints=10)
    
    framework.run_optimization()


if __name__ == "__main__":
    params = sys.argv
    # thcTest()
    # thcTest2(params[1], params[2], params[3], params[4], params[5])

    
    
    # substr1 = ['#!/bin/bash',
    #             '##NECESSARY JOB SPECIFICATIONS',
    #             '#BSUB -J sub{0}',
    #             '#BSUB -P 082824066694',
    #             '#BSUB -L /bin/bash',
    #             '#BSUB -W 01:00',
    #             '#BSUB -n 20',
    #             '#BSUB -R "span[ptile=20]"',
    #             '#BSUB -R "rusage[mem=2560]"',
    #             '#BSUB -M 2560',
    #             '#BSUB -o LSFOut/Out.%J',
    #             '#BSUB -e LSFOut/Err.%J',
    #             '#BSUB -u richardcouperthwaite@tamu.edu',
    #             '#BSUB -N',
    #             'cd $SCRATCH/BAREFOOT',
    #             '#module load Python/3.6.6-intel-2018b',
    #             'module load PyTorch/1.1.0-foss-2019a-Python-3.7.2',
    #             'source venv/bin/activate',
    #             'cd barefootTest',
    #             'python subProcess.py {0}']
    
    # substr2 = ['#!/bin/bash',
    #             'cd /scratch/user/richardcouperthwaite/BAREFOOT/barefootTest/subprocess',
    #             'bsub < {0}.sh']
    
    # with open("data/processStrings", 'wb') as f:
    #     dump(['\n'.join(substr1), '\n'.join(substr2)], f)
    
    
    tc_gp = TC_GP()
    with open("data/tc_gpObj", 'wb') as f:
        dump(tc_gp, f)
    
    # mechModelTest()
    mechModelTest2(params[1], params[2], params[3], params[4], params[5])
    