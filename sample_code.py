# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:23:37 2021

@author: Richard Couperthwaite
"""

import numpy as np
from pickle import dump
from barefoot import barefoot
from sys import argv
from time import sleep
from pyDOE import lhs

def rom1(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return -80*np.sin(2*x[:,0])/(441*np.pi) - 160*np.sin(4*x[:,0])/(81*np.pi) + np.pi*np.sin(5*x[:,0])/2 - 48*np.sin(x[:,1])/(1225*np.pi) - 16*np.sin(3*x[:,1])/(81*np.pi) - 240*np.sin(5*x[:,1])/(121*np.pi)

def rom2(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return -60*np.sin(2.5*x[:,0])/(441*np.pi) - 60*np.sin(0.5*x[:,1])/(1200*np.pi) - 160*np.sin(4.2*x[:,0])/(81*np.pi) + np.pi*np.sin(5.3*x[:,0])/2 - 16*np.sin(3.2*x[:,1])/(81*np.pi) - 240*np.sin(5.3*x[:,1])/(121*np.pi) 

def rom3(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return -100*np.sin(1.5*x[:,0])/(400*np.pi) - 36*np.sin(1.5*x[:,1])/(1200*np.pi) - 160*np.sin(4.5*x[:,0])/(81*np.pi) + np.pi*np.sin(5.5*x[:,0])/2 - 16*np.sin(3.5*x[:,1])/(81*np.pi) - 240*np.sin(5.5*x[:,1])/(121*np.pi)

def truth(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return np.abs(x[:,0])*np.sin(5*x[:,0]) + np.abs(x[:,1])*np.sin(6*x[:,1])

def runBatch():
    ROMList = [rom1, rom2, rom3]
    framework = barefoot(ROMModelList=ROMList, 
                         TruthModel=truth, 
                         calcInitData=True, 
                         initDataPathorNum=[2,2,2,2], 
                         multiNode=0, 
                         workingDir=".", 
                         calculationName="BatchOnly", 
                         nDim=2, 
                         input_resolution=5, 
                         restore_calc=False,
                         updateROMafterTM=False, 
                         externalTM=False, 
                         acquisitionFunc="EI",
                         A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], 
                         keepSubRunning=True, 
                         verbose=False, 
                         sampleScheme="LHS", 
                         tmSampleOpt="EI", 
                         logname="BAREFOOT",
                         maximize=True, 
                         train_func=[], 
                         reification=False, 
                         batch=True)

    modelParam = {'model_l': [[0.1608754, 0.2725361],
                                  [0.17094462, 0.28988983],
                                  [0.10782092, 0.18832378]],
                          'model_sf': [6.95469898,8.42299498,2.98009081],
                          'model_sn': [0.05, 0.05, 0.05],
                          'means': [-2.753353101070388e-16,-1.554312234475219e-16,4.884981308350689e-17],
                          'std': [1.2460652976290285,1.3396622409903254,1.3429644403939915],
                          'err_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                          'err_sf': [1,1,1],
                          'err_sn': [0.01, 0.01, 0.01],
                  'costs': [1,1,1,1]}
    
    framework.initialize_parameters(modelParam=modelParam, 
                                    covFunc="M32", 
                                    iterLimit=10,  
                                    sampleCount=50, 
                                    hpCount=1000, 
                                    batchSize=10, 
                                    tmIter=5, 
                                    upperBound=1, 
                                    lowBound=0.0001, 
                                    fusedPoints=10, 
                                    fusedHP=[3, 0.1, 0.1])

    framework.run_optimization()
    
    
def runReifi():
    initx = lhs(2,8)
    
    init1 = initx[0:2,:]
    init2 = initx[2:4,:]
    init3 = initx[4:6,:]
    initTM = initx[6:8,:]
    
    y1 = rom1(init1)
    y2 = rom2(init2)
    y3 = rom3(init3)
    tm = truth(initTM)
    
    datadict = {"TMInitOutput": tm, 
                "TMInitInput": initTM, 
                "ROMInitOutput": [y1, y2, y3], 
                "ROMInitInput": [init1, init2, init3]}
    
    with open("data/initData", 'wb') as f:
        dump(datadict, f)
    
    ROMList = [rom1, rom2, rom3]
    framework = barefoot(ROMModelList=ROMList, 
                         TruthModel=truth, 
                         calcInitData=True, 
                         initDataPathorNum=[2,2,2,2], 
                         multiNode=0, 
                         workingDir=".", 
                         calculationName="ReificationOnly", 
                         nDim=2, 
                         input_resolution=5, 
                         restore_calc=False,
                         updateROMafterTM=False, 
                         externalTM=False, 
                         acquisitionFunc="EI",
                         A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], 
                         keepSubRunning=True, 
                         verbose=False, 
                         sampleScheme="LHS", 
                         tmSampleOpt="EI", 
                         logname="BAREFOOT",
                         maximize=True, 
                         train_func=[], 
                         reification=False, 
                         batch=True)

    modelParam = {'model_l': [[0.1608754, 0.2725361],
                                  [0.17094462, 0.28988983],
                                  [0.10782092, 0.18832378]],
                          'model_sf': [6.95469898,8.42299498,2.98009081],
                          'model_sn': [0.05, 0.05, 0.05],
                          'means': [-2.753353101070388e-16,-1.554312234475219e-16,4.884981308350689e-17],
                          'std': [1.2460652976290285,1.3396622409903254,1.3429644403939915],
                          'err_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                          'err_sf': [1,1,1],
                          'err_sn': [0.01, 0.01, 0.01],
                  'costs': [1,1,1,1]}
    
    framework.initialize_parameters(modelParam=modelParam, 
                                    covFunc="M32", 
                                    iterLimit=10,  
                                    sampleCount=50, 
                                    hpCount=1000, 
                                    batchSize=10, 
                                    tmIter=5, 
                                    upperBound=1, 
                                    lowBound=0.0001, 
                                    fusedPoints=10, 
                                    fusedHP=[3, 0.1, 0.1])

    framework.run_optimization()



def runBAREFOOT():
    substr1 = ['#!/bin/bash',
               '##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION',
               '#SBATCH --export=NONE                #Do not propagate environment',
               '#SBATCH --get-user-env=L             #Replicate login environment',
               ' ',
               '##NECESSARY JOB SPECIFICATIONS',
               '#SBATCH --job-name=sub{0}',
               '#SBATCH --time=10:00:00',
               '#SBATCH --nodes=1            ',       
               '#SBATCH --ntasks-per-node=48    ',
               '#SBATCH --mem=360G',
               '#SBATCH --output=Out.%j',
               '',
               '##OPTIONAL JOB SPECIFICATIONS',
               '',
               'ml GCCcore/10.2.0 Python/3.8.6',
               'cd $SCRATCH/BAREFOOT',
               'source venv/bin/activate',
               'cd barefootTest',
                'python subProcess.py {0}']
    
    substr2 = ['#!/bin/bash',
                'cd $SCRATCH/BAREFOOT/barefootTest/subprocess',
                'sbatch {0}.sh']
    
    with open("data/processStrings", 'wb') as f:
        dump(['\n'.join(substr1), '\n'.join(substr2)], f)
    
    ROMList = [rom1, rom2, rom3]
    framework = barefoot(ROMModelList=ROMList, 
                         TruthModel=truth, 
                         calcInitData=True, 
                         initDataPathorNum=[2,2,2,2], 
                         multiNode=5, 
                         workingDir=".", 
                         calculationName="FullBarefoot", 
                         nDim=2, 
                         input_resolution=5, 
                         restore_calc=False,
                         updateROMafterTM=False, 
                         externalTM=False, 
                         acquisitionFunc="EI",
                         A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], 
                         keepSubRunning=True, 
                         verbose=False, 
                         sampleScheme="LHS", 
                         tmSampleOpt="EI", 
                         logname="BAREFOOT",
                         maximize=True, 
                         train_func=[], 
                         reification=False, 
                         batch=True)

    modelParam = {'model_l': [[0.1608754, 0.2725361],
                                  [0.17094462, 0.28988983],
                                  [0.10782092, 0.18832378]],
                          'model_sf': [6.95469898,8.42299498,2.98009081],
                          'model_sn': [0.05, 0.05, 0.05],
                          'means': [-2.753353101070388e-16,-1.554312234475219e-16,4.884981308350689e-17],
                          'std': [1.2460652976290285,1.3396622409903254,1.3429644403939915],
                          'err_l': [[0.1,0.1],[0.1,0.1],[0.1,0.1]],
                          'err_sf': [1,1,1],
                          'err_sn': [0.01, 0.01, 0.01],
                  'costs': [1,1,1,1]}
    
    framework.initialize_parameters(modelParam=modelParam, 
                                    covFunc="M32", 
                                    iterLimit=10,  
                                    sampleCount=10, 
                                    hpCount=1000, 
                                    batchSize=10, 
                                    tmIter=5, 
                                    upperBound=1, 
                                    lowBound=0.0001, 
                                    fusedPoints=10, 
                                    fusedHP=[3, 0.1, 0.1])

    framework.run_optimization()




if __name__ == "__main__":
    # Uncomment the function that you wish to run
    runBatch()
    #runReifi()
    #runBAREFOOT()