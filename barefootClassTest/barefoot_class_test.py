# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:11:50 2021

@author: Richard Couperthwaite
"""

import os
from pickle import load, dump
import subprocess
from time import time, sleep
from shutil import rmtree
import numpy as np
import pandas as pd
from reificationFusion import model_reification
from pyDOE import lhs
import concurrent.futures
from multiprocessing import cpu_count
from kmedoids import kMedoids
from scipy.spatial import distance_matrix
from copy import deepcopy
from acquisitionFunc import expected_improvement, knowledge_gradient
import logging

# create logger with 'spam_application'
logger = logging.getLogger('BAREFOOT')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('BAREFOOT.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(fh)

def k_medoids(sample, num_clusters):
    D = distance_matrix(sample, sample)
    M, C = kMedoids(D, num_clusters)
    return M, C  

def call_model(param):
    output = param["Model"](param["Input Values"])
    return output

def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape

def apply_constraints(samples, ndim, resolution=[], A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[]):
    sampleSelection = True
    constraints = np.array((5))
    if A != []:
        constraints[0] = 1
    if Aeq != []:
        constraints[1] = 1
    if lb != []:
        constraints[2] = 1
    if ub != []:
        constraints[3] = 1
    if func != []:
        if (type(func) == list):
            constraints[4] = len(func)
        else:
            constraints[4] = 1
    lhs_samples = samples
    while sampleSelection:
        x = lhs(ndim, lhs_samples)
        if resolution != []:
            x = np.round(x, decimals=resolution)
        constr_check = np.zeros((lhs_samples, ndim))
        
        # Apply inequality constraints
        if (A != []) and (b != []) and (len(A) == ndim):
            A_tile = np.tile(np.array(A), (lhs_samples,1))
            constr_check += A_tile*x <= b
            constraints[0] = 0

        # Apply equality constraints
        if (Aeq != []) and (beq != []):
            Aeq_tile = np.tile(np.array(Aeq), (lhs_samples,1))
            constr_check += Aeq_tile*x <= beq
            constraints[1] = 0
        
        # Apply Lower and Upper Bounds
        if (lb != []) and (len(lb) == ndim):
            lb_tile = np.tile(np.array(lb).reshape((1,ndim)), (lhs_samples,1))
            constr_check += x < lb_tile
            constraints[2] = 0
        if (ub != []) and (len(ub) == ndim):
            ub_tile = np.tile(np.array(ub).reshape((1,ndim)), (lhs_samples,1))
            constr_check += x > ub_tile
            constraints[3] = 0
        
        constr_check = np.sum(constr_check, axis=1)
        
        # Apply custom function constraints
        if (type(func) == list) and (func != []):
            for ii in range(len(func)):
                try:
                    constr_check += func[ii](x)
                    constraints[4] -= 0
                except:
                    pass
        elif (type(func) != list) and (func != []):
            try:
                constr_check += func(x)    
                constraints[4] = 0
            except:
                pass
        index = np.where(constr_check == 0)[0]
        
        if len(index) >= samples:
            x = x[index,:]
            sampleSelection = False
            if np.sum(constraints) != 0:
                const_satisfied = False
            else:
                const_satisfied = True
        else:
            if lhs_samples/samples < 2000:
                lhs_samples += samples
            else:
                x = lhs(ndim, samples)
                sampleSelection = False
                const_satisfied = False
    return x, const_satisfied

def calculate_KG(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        knowledge gradient of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum knowledge gradient value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum knowledge gradient and the maximum of the fused model

    """
    (finish, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output    
    output[0] = np.max(fused_mean)
    # Calculate the knowledge gradient for all test point
    nu_star, x_star, NU = knowledge_gradient(true_sample_count, 
                                              0.1, 
                                              fused_mean, 
                                              fused_var)
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return output

def calculate_EI(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        expected improvement of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum expected improvement and the maximum of the fused model

    """
    (finish, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output  
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Calculate the expected improvement for all test point
    nu_star, x_star, NU = expected_improvement(curr_max, 
                                               0.01, 
                                               fused_mean, 
                                               fused_var)
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return output   

def fused_calculate(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        maximum of a fused model generated from a reification object.

    Returns
    -------
    results : list
        The output from the module contains the maximum of the fused model as 
        well as the index of the test point that corresponds with that value.

    """
    (finish, model_temp, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi) = param
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Predict the mean and variance at each test point
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    # Find the maximum of the fused model
    index_max = np.nonzero(fused_mean == np.max(fused_mean))
    # return the maximum value and the index of the test point that corresponds
    # with the maximum value
    return [np.max(fused_mean),index_max[0][0]]

class barefoot():
    def __init__(self, ROMModelList=[], TruthModel=[], calcInitData=True, 
                 initDataPathorNum=[], multiNode=0, workingDir=".", 
                 calculationName="Calculation", nDim=1, input_resolution=5, restore_calc=False,
                 updateROMafterTM=False, externalTM=False, acquisitionFunc="KG",
                 A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], keepSubRunning=True):
        """
        Python Class for Batch Reification/Fusion Optimization (BAREFOOT) Framework Calculations

        Parameters
        ----------
        ROMModelList       : This is the list of functions that are the cheap information sources.
                             These need to be in a form that ensures that by providing the unit hypercube
                             input, the function will provide the required output
        TruthModel         : This is the Truth model, or the function that needs to be optimized.
        calcInitData       : This variable controls whether the initial data is calculated for
                             each of the models or is retrieved from a file
        initDataPathorNum  : This variable holds the number of initial datapoints to evaluate for each
                             information source (including the Truth Model), or, when initial data is 
                             loaded from a file, holds the path to the initial data file
        multiNode          : This variable reflects the number of subprocesses that will be used
                             for the calculations. A value of zero indicates all calculations will
                             be completed on the main compute node.
        workingDir         : This is the path to the working directory. In some cases it may be desirable
                             to store data separately from the code, this will allow the data to be stored
                             in alternate locations. Can also be used if the relative directory reference
                             is not working correctly.
        calculationName    : This is the name for the calculation and will change the results directory name
        nDim               : The number of dimensions for the input space that will be used
        restore_calc       : This parameter toggles whether the framework data is set up from the information
                             provided or retrieved from a save_state file. This can be used to restart a calculation
        updateROMafterTM   : This parameter allows the reduced order models to be retrained after getting more data
                             from the Truth Model. The model function calls do not change, so the training needs to 
                             reflect in the same function.
        externalTM         : In cases where it is necessary to evaluate the Truth Model separate to the
                             framework (for example, if the Truth Model is an actual experiment), this toggles
                             the output of the predicted points to a separate file for use externally. The
                             framework is shut down after the data is output, see test examples for how to restart
                             the framework after the external Truth Model has been evaluated
        acquisitionFunc    : The acquisition function to use to evaluate the next best points for the reduced
                             order models. Currently the options are "KG" for Knowledge Gradient and "EI" for expected
                             improvement.
        A, b, Aeq, beq     : Equality and inequality constraints according to the following equations:
                             1) A*x <= b
                             2) Aeq*x == b
        ub, lb             : Upper bounds and lower bounds for inputs, all inputs must receive a value
                             (Specify 0 for lb and 1 for ub if there is no bound for that input)
        func               : function constraints, must take the input matrix (x) and output a vector of length
                             equal to the number of samples in the input matrix (x) with boolean values.
                             
        """
        logger.info("Start BAREFOOT Framework Initialization")
        
        # Restore a previous calculation and restart the timer or load new
        # information and initialize
        
        if restore_calc:
            self.__load_from_save__()
            self.timeCheck = time()
            logger.info("Previous Save State Restored")
        else:
            self.timeCheck = time()
            self.ROM = ROMModelList
            self.TM = TruthModel
            self.TMInitInput = []
            self.TMInitOutput = []
            self.ROMInitInput = []
            self.ROMInitOutput = []
            self.inputLabels = []
            self.multinode = multiNode
            self.workingDir = workingDir
            self.calculationName = calculationName
            self.calcInitData = calcInitData
            self.initDataPathorNum = initDataPathorNum
            self.currentIteration = -1
            self.nDim = nDim
            self.res = input_resolution
            self.A = A
            self.b = b
            self.Aeq = Aeq
            self.beq = beq
            self.ub = ub
            self.lb = ub
            self.constr_func
            self.updateROMafterTM = updateROMafterTM
            self.externalTM = externalTM
            self.acquisitionFunc = acquisitionFunc
            self.__create_dir_and_files()
            self.__create_output_dataframes__()
            self.__get_initial_data__()
            logger.info("Initialization Completed")  
        
    
    def __create_dir_and_files(self):
        # Create the required directories for saving the results and the subprocess
        # information if applicable
        try:
            os.mkdir('{}/results'.format(self.workingDir))
            logger.debug("Results Directory Created Successfully")
        except FileExistsError:
            logger.debug("Results Directory Already Exists")
            pass
        try:
            os.mkdir('{}/results/{}'.format(self.workingDir, 
                                            self.calculationName))
            logger.debug("Calculation Results Directory [{}] Created Successfully".format(self.calculationName))
        except FileExistsError:
            logger.debug("Calculation Results Directory [{}] Already Exists".format(self.calculationName))
            pass
        # If using subprocesses, create the folder structure needed
        if self.multinode != 0:
            if os.path.exists('{}/subprocess'.format(self.workingDir)):
                rmtree('{}/subprocess'.format(self.workingDir))
                logger.debug("Existing Subprocess Directory Removed")
            os.mkdir('{}/subprocess'.format(self.workingDir))
            os.mkdir('{}/subprocess/LSFOut'.format(self.workingDir))
            logger.debug("Subprocess Directory Created")
                
    def __create_output_dataframes__(self):
        # The output of the framework is contained in two pandas dataframes
        # the evaluatedPoints df contains all the points that have been 
        # evaluated from all models
        labels1 = ["Model Index", "Iteration", "y"]
        for ii in range(self.nDim):
            labels1.append("x{}".format(ii))
            self.inputLabels.append("x{}".format(ii))
        self.evaluatedPoints = pd.DataFrame(columns=labels1)
        # the iterationData df
        labels2 = ["Iteration", "Max Found", "Calculation Time", "Truth Model"]
        for ii in range(len(self.ROM)):
            labels2.append("ROM {}".format(ii))
        self.iterationData = pd.DataFrame(columns=labels2)
        logger.debug("Output Dataframes Created")
    
    def __save_output_dataframes__(self):
        with open('{}/results/{}/evaluatedPoints'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(self.evaluatedPoints, f)
        self.evaluatedPoints.to_csv('{}/results/{}/evaluatedPoints.csv'.format(self.workingDir, self.calculationName))
        with open('{}/results/{}/iterationData'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(self.iterationData, f)
        self.iterationData.to_csv('{}/results/{}/iterationData.csv'.format(self.workingDir, self.calculationName))
        logger.info("Dataframes Pickled and Dumped to Results Directory")
            
    def __save_calculation_state__(self):
        with open('{}/data/{}_save_state'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(self, f)
        logger.info("Calculation State Saved")
        
    def __load_from_save__(self):
        try:
            print('{}/data/{}_save_state'.format(self.workingDir, self.calculationName))
            with open('{}/data/{}_save_state'.format(self.workingDir, self.calculationName), 'rb') as f:
                saveState = load(f)
                logger.debug("Save State File Found")
            for item in vars(saveState).items():
                setattr(self, item[0], item[1])
        except FileNotFoundError:
            self.loadFailed = True
            logger.warning("Could not find Save State File")
        
    def __add_to_evaluatedPoints(self, modelIndex, eval_x, eval_y):
        temp = np.zeros((eval_x.shape[0], self.nDim+3))
        temp[:,0] = modelIndex
        temp[:,1] = self.currentIteration
        temp[:,2] = eval_y
        temp[:,3:] = eval_x
        temp = pd.DataFrame(temp, columns=self.evaluatedPoints.columns)
        self.evaluatedPoints = pd.concat([self.evaluatedPoints,temp])
        logger.debug("{} New Points Added to Evaluated Points Dataframe".format(eval_x.shape[0]))
        
    def __add_to_iterationData(self, calcTime, iterData):
        temp = np.zeros((1,4+len(self.ROM)))
        temp[0,0] = self.currentIteration
        temp[0,1] = self.maxTM
        temp[0,2] = calcTime
        temp[0,3] = iterData[-1]
        temp[0,4:] = iterData[0:len(self.ROM)]
        temp = pd.DataFrame(temp, columns=self.iterationData.columns)
        self.iterationData = pd.concat([self.iterationData,temp])
        logger.debug("Iteration {} Data saved to Dataframe".format(self.currentIteration))
        
    def __get_initial_data__(self):
        params = []
        count = []
        param_index = 0
        self.maxTM = -np.inf
        if self.calcInitData:
            logger.debug("Start Calculation of Initial Data")
            for ii in range(len(self.ROM)):
                count.append(0)                
                initInput, check = apply_constraints(self.initDataPathorNum[ii], 
                                                     self.nDim, self.res,
                                                      self.A, self.b, self.Aeq, self.beq, 
                                                      self.lb, self.ub, self.constr_func)
                if check:
                    logger.debug("Initial Data - All constraints applied successfully")
                else:
                    logger.critical("Initial Data - Some or All Constraints Could not Be applied! Continuing Without Constraints")
                
                for jj in range(self.initDataPathorNum[ii]):
                    params.append({"Model Index":ii,
                                   "Model":self.ROM[ii],
                                   "Input Values":initInput[jj,:],
                                   "ParamIndex":param_index})
                    param_index += 1
                self.ROMInitInput.append(np.zeros_like(initInput))
                self.ROMInitOutput.append(np.zeros(self.initDataPathorNum[ii]))
            count.append(0)
            
            initInput, check = apply_constraints(self.initDataPathorNum[ii+1], 
                                                     self.nDim, self.res,
                                                      self.A, self.b, self.Aeq, self.beq, 
                                                      self.lb, self.ub, self.constr_func)
            if check:
                logger.debug("Initial Data - All constraints applied successfully")
            else:
                logger.critical("Initial Data - Some or All Constraints Could not Be applied! Continuing Without Constraints")
            for jj in range(self.initDataPathorNum[-1]):
                params.append({"Model Index":-1,
                               "Model":self.TM,
                               "Input Values":initInput[jj,:],
                               "ParamIndex":param_index})
                param_index += 1
            self.TMInitInput = np.zeros_like(initInput)
            self.TMInitOutput = np.zeros(self.initDataPathorNum[-1])
            
            temp_x = np.zeros((len(params), self.nDim))
            temp_y = np.zeros(len(params))
            temp_index = np.zeros(len(params))
            logger.debug("Parameters Defined. Starting Concurrent.Futures Calculation")
            with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                for result_from_process in zip(params, executor.map(call_model, params)):
                    par, results = result_from_process
                    if par["Model Index"] != -1:
                        self.ROMInitInput[par["Model Index"]][count[par["Model Index"]],:] = par["Input Values"]
                        self.ROMInitOutput[par["Model Index"]][count[par["Model Index"]]] = results
                        temp_x[par["ParamIndex"],:] = par["Input Values"]
                        temp_y[par["ParamIndex"]] = results
                        temp_index[par["ParamIndex"]] = par["Model Index"]
                    else:
                        self.TMInitInput[count[par["Model Index"]],:] = par["Input Values"]
                        self.TMInitOutput[count[par["Model Index"]]] = results
                        if results > self.maxTM:
                            self.maxTM = results
                        temp_x[par["ParamIndex"],:] = par["Input Values"]
                        temp_y[par["ParamIndex"]] = results
                        temp_index[par["ParamIndex"]] = par["Model Index"]
                    count[par["Model Index"]] += 1
            logger.debug("Concurrent.Futures Calculation Completed")
        else:
            logger.debug("Start Loading Initial Data from Files")
            with open(self.initDataPathorNum, 'rb') as f:
                data = load(f)
            self.TMInitOutput = data["TMInitOutput"]
            self.TMInitInput = data["TMInitInput"]
            self.ROMInitOutput = data["ROMInitOutput"]
            self.ROMInitInput = data["ROMInitInput"]
            
            temp_x = np.zeros((self.TMInitOutput.shape[0]+self.ROMInitOutput.shape[0], 
                               self.nDim))
            temp_y = np.zeros(self.TMInitOutput.shape[0]+self.ROMInitOutput.shape[0])
            temp_index = np.zeros(self.TMInitOutput.shape[0]+self.ROMInitOutput.shape[0])
            
            ind = 0
            
            for ii in range(len(self.ROM)):
                for jj in range(self.ROMInitOutput[ii].shape[0]):
                    temp_x[ind,:] = self.ROMInitInput[ii][jj,:]
                    temp_y[ind] = self.ROMInitOutput[ii][jj]
                    temp_index[ind] = ii
                    ind += 1
                count.append(self.ROMInitInput[ii].shape[0])
            for jj in range(self.TMInitOutput.shape[0]):
                temp_x[ind,:] = self.TMInitInput[jj,:]
                temp_y[ind] = self.TMInitOutput[jj]
                temp_index[ind] = -1
                ind += 1
            count.append(self.TMInitInput.shape[0])
            logger.debug("Loading Data From File Completed")
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.__add_to_iterationData(time()-self.timeCheck, np.array(count))
        logger.debug("Initial Data Saved to Dataframes")
        self.timeCheck = time()
    
    def initialize_parameters(self, modelParam, covFunc="M32", iterLimit=100,  
                              sampleCount=50, hpCount=100, batchSize=5, 
                              tmIter=10, totalBudget=1e6, tmBudget=50000, 
                              upperBound=1, lowBound=0.0001, fusedPoints=5):
        logger.debug("Start Initializing Reification Object Parameters")
        self.covFunc = covFunc 
        self.iterLimit = iterLimit 
        self.sampleCount = sampleCount 
        self.hpCount = hpCount 
        self.batchSize = batchSize
        self.tmIterLim = tmIter 
        self.totalBudget = totalBudget
        self.tmBudget = tmBudget
        self.upperBound = upperBound
        self.lowBound = lowBound
        self.modelParam = modelParam
        self.modelCosts = modelParam["costs"]
        if self.upperBound > 1:
            midway = (self.hpCount - (self.hpCount % 2))/2
            lower = np.linspace(self.lowBound, 1.0, num=int(midway), endpoint=False)
            upper = np.linspace(1.0, self.upperBound, num=int(midway)+int(self.hpCount % 2), endpoint=True)
            all_HP = np.append(lower, upper)
        else:
            all_HP = np.linspace(self.lowBound, self.upperBound, num=self.hpCount, endpoint=True)
        self.fusedModelHP = np.zeros((self.hpCount,self.nDim+1))
        for i in range(self.hpCount):
            for j in range(self.nDim+1):
                self.fusedModelHP[i,j] = all_HP[np.random.randint(0,self.hpCount)]
        temp = np.linspace(0,1,num=fusedPoints)
        arr_list = []
        for ii in range(self.nDim):
            arr_list.append(temp)
        self.xFused = cartesian(*arr_list)
        logger.debug("Create Reification Object")
        self.reificationObj = model_reification(self.ROMInitInput, self.ROMInitOutput, 
                                          self.modelParam['model_l'], 
                                          self.modelParam['model_sf'], 
                                          self.modelParam['model_sn'], 
                                          self.modelParam['means'], 
                                          self.modelParam['std'], 
                                          self.modelParam['err_l'], 
                                          self.modelParam['err_sf'], 
                                          self.modelParam['err_sn'], 
                                          self.TMInitInput, self.TMInitOutput, 
                                          len(self.ROM), self.nDim, self.covFunc)
        self.allTMInput = []
        self.allTMOutput = []
        self.tmBudgetLeft = self.tmBudget
        self.totalBudgetLeft = self.totalBudget
        self.currentIteration += 1
        self.tmIterCount = 0
        logger.info("Reification Object Initialized. Ready for Calculations")
        
    def run_single_node(self):
        logger.info("Start Single Node Calculation")
        while True:
            self.timeCheck = time()
            
            x_test, check = apply_constraints(self.sampleCount, 
                                              self.nDim, self.res,
                                              self.A, self.b, self.Aeq, self.beq, 
                                              self.lb, self.ub, self.constr_func)
            if check:
                logger.debug("Single Node - All constraints applied successfully")
            else:
                logger.critical("Single Node - Some or All Constraints Could not Be applied! Continuing Without Constraints")
            
            
            new_mean = []
            for iii in range(len(self.ROM)):
                new, var = self.reificationObj.predict_low_order(x_test, iii)
                new_mean.append(new)
                    
            kg_output = [] 
            parameters = []
            count = 0
            logger.debug("Set Up Parameters for Acquisition Function Evaluation")
            for jj in range(len(self.ROM)):
                for kk in range(self.sampleCount):
                    model_temp = deepcopy(self.reificationObj)
                    model_temp.update_GP(np.expand_dims(x_test[kk], axis=0), 
                                          np.expand_dims(np.array([new_mean[jj][kk]]), 
                                                    axis=0), jj)
    
                    for mm in range(self.hpCount):
                        parameters.append((1, model_temp, self.xFused, self.fusedModelHP[mm,:],
                                        self.covFunc, x_test, jj, kk, mm, self.sampleCount,
                                        self.modelParam['costs'], self.maxTM))
                        count += 1
                        
    
            kg_output = []
            logger.info("Start Acquisition Function Evaluations for {} Parameter Sets".format(len(parameters)))
            with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                for result_from_process in zip(parameters, executor.map(calculate_EI,parameters)):
                    params, results = result_from_process
                    kg_output.append(results)
            logger.info("Acquisition Function Evaluations Completed")
            
            medoid_out = self.__kg_calc_clustering(kg_output)
            
            params = []
            count = np.zeros((len(self.ROM)+1)) 
            current = np.array(self.iterationData.iloc[:,3:])[-1,:]
            count[0:len(self.ROM)] = current[1:]
            count[-1] = current[0]
            param_index = 0
            logger.debug("Define Parameters for ROM Function Evaluations")
            for iii in range(medoid_out.shape[0]):
                x_index = 6 + self.nDim
                params.append({"Model Index":medoid_out[iii,3],
                               "Model":self.ROM[medoid_out[iii,3]],
                               "Input Values":np.array(medoid_out[iii,x_index:], dtype=np.float),
                               "ParamIndex":param_index})
                param_index += 1

            temp_x = np.zeros((len(params), self.nDim))
            temp_y = np.zeros(len(params))
            temp_index = np.zeros(len(params)) 
            costs = np.zeros(len(params))
            logger.info("Start ROM Function Evaluations | {} Calculations".format(len(params)))
            with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                for result_from_process in zip(params, executor.map(call_model, params)):
                    par, results = result_from_process
                    self.reificationObj.update_GP(par["Input Values"], results, par["Model Index"])
                    temp_x[par["ParamIndex"],:] = par["Input Values"]
                    temp_y[par["ParamIndex"]] = results
                    temp_index[par["ParamIndex"]] = par["Model Index"]
                    costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                    count[par["Model Index"]] += 1
            self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
            self.totalBudgetLeft -= np.sum(costs)
            self.tmBudgetLeft -= np.sum(costs)
            logger.info("ROM Function Evaluations Completed")
            
            if (self.tmBudgetLeft < 0) or (self.tmIterCount == self.tmIterLim):
                logger.info("Start Truth Model Evaluations")
                self.tmIterCount = 0
                self.tmBudgetLeft = self.tmBudget
                
                # create a test set that is dependent on the number of dimensions            
                tm_test, check = apply_constraints(2500*self.nDim, 
                                              self.nDim, self.res,
                                              self.A, self.b, self.Aeq, self.beq, 
                                              self.lb, self.ub, self.constr_func)
                if check:
                    logger.debug("Single Node - All constraints applied successfully")
                else:
                    logger.critical("Single Node - Some or All Constraints Could Not Be Applied! Continuing Without Constraints")
                
                parameters = []
                
                # initialize the parameters for the fused model calculations and
                # start the calculation
                logger.debug("Define Parameters for Max Value Evaluations")
                for mm in range(self.hpCount):
                    parameters.append((1, self.reificationObj, self.xFused, self.fusedModelHP[mm,:],
                                    self.covFunc, tm_test, self.maxTM, 0.01))
                
                fused_output = []
                logger.info("Start Max Value Calculations | {} Sets".format(len(parameters)))
                with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                    for result_from_process in zip(parameters, executor.map(fused_calculate,parameters)):
                        params, results = result_from_process
                        fused_output.append(results)
                logger.info("Max Value Calculations Completed")
            
                # cluster the output from the fused model calculations
                fused_output = np.array(fused_output, dtype=object)
                try:
                    medoids, clusters = k_medoids(fused_output, self.batchSize)
                except:
                    if (self.nDim > 1) and (self.nDim < self.batchSize):
                        try:
                            medoids, clusters = k_medoids(fused_output, self.nDim)
                        except:
                            medoids, clusters = k_medoids(fused_output, 1)
                    else:
                        medoids, clusters = k_medoids(fused_output, 1)
                # Calculate the new Truth Model values and add them to the data
                params = []
                count = np.zeros((len(self.ROM)+1)) 
                current = np.array(self.iterationData.iloc[:,3:])[-1,:]
                count[0:len(self.ROM)] = current[1:]
                count[-1] = current[0]
                param_index = 0
                logger.debug("Define Parameters for Truth Model Evaluations")
                for iii in range(len(medoids)):
                    x_index = 6 + self.nDim
                    params.append({"Model Index":-1,
                                   "Model":self.TM,
                                   "Input Values":np.array(tm_test[fused_output[medoids[iii],1],:], dtype=np.float),
                                   "ParamIndex":param_index})
                    param_index += 1
                
                
                if self.externalTM:
                    self.external_TM_data_save(params, count)
                    break
                else:
                    temp_x = np.zeros((len(params), self.nDim))
                    temp_y = np.zeros(len(params))
                    temp_index = np.zeros(len(params)) 
                    costs = np.zeros(len(params))
                    logger.info("Start Truth Model Evaluations | {} Sets".format(len(params)))
                    with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                        for result_from_process in zip(params, executor.map(call_model, params)):
                            par, results = result_from_process
                            self.reificationObj.update_truth(par["Input Values"], results)
                            temp_x[par["ParamIndex"],:] = par["Input Values"]
                            temp_y[par["ParamIndex"]] = results
                            temp_index[par["ParamIndex"]] = par["Model Index"]
                            costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                            count[par["Model Index"]] += 1
                    logger.info("Truth Model Evaluations Completed")
                    self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
                    self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
                    if np.max(temp_y) > self.maxTM:
                        self.maxTM = np.max(temp_y)
            
            self.__add_to_iterationData(time()-self.timeCheck, count)
            self.timeCheck = time()
            
            if self.updateROMafterTM:
                self.__update_reduced_order_models__()
            
            self.__save_output_dataframes__()
            self.__save_calculation_state__()
            logger.info("Iteration {} Completed Successfully".format(self.currentIteration))
            
            if (self.totalBudgetLeft < 0) or (self.currentIteration >= self.iterLimit):
                logger.info("Iteration or Budget Limit Met or Exceeded | Single Node Calculation Completed")
                break
            
            self.tmIterCount += 1
            self.currentIteration += 1
    
        
    def __kg_calc_clustering(self, kg_output):
        # convert to a numpy array for ease of indexing
        kg_output = np.array(kg_output, dtype=object)
        # print(kg_output)
        # print(kg_output.shape)
        point_selection = {}
        logger.debug("Extract Points for Clustering from Acquisition Function Evaluations")
        for iii in range(kg_output.shape[0]):
            # print(iii)
            try:
                if kg_output[iii,3] in point_selection[kg_output[iii,2]]['models']:
                    if kg_output[iii,1] > point_selection[kg_output[iii,2]]['nu'][kg_output[iii,3]]:
                        point_selection[kg_output[iii,2]]['nu'][kg_output[iii,3]] = kg_output[iii,1]
                        point_selection[kg_output[iii,2]]['kg_out'][kg_output[iii,3]] = iii
                else:
                    point_selection[kg_output[iii,2]]['models'].append(kg_output[iii,3])
                    point_selection[kg_output[iii,2]]['nu'][kg_output[iii,3]] = kg_output[iii,1]
                    point_selection[kg_output[iii,2]]['kg_out'][kg_output[iii,3]] = iii
            except KeyError:
                point_selection[kg_output[iii,2]] = {'models':[kg_output[iii,3]],
                                                     'nu':[],
                                                     'kg_out':[]}
                for mm in range(len(self.ROM)):
                    point_selection[kg_output[iii,2]]['nu'].append(1e-6)
                    point_selection[kg_output[iii,2]]['kg_out'].append(-1)
                point_selection[kg_output[iii,2]]['nu'][kg_output[iii,3]] = kg_output[iii,1]
                point_selection[kg_output[iii,2]]['kg_out'][kg_output[iii,3]] = iii
        
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
        logger.debug("Cluster Acquistion Function Evaluations")
        if med_input.shape[0] > self.batchSize:
            medoids, clusters = k_medoids(med_input[:,0:3], self.batchSize)
        else:
            medoids, clusters = k_medoids(med_input[:,0:3], 1)       
        
        # next, need to get the true values for each of the medoids and update the
        # models before starting next iteration.
        logger.debug("Extract True Values for Medoids")
        medoid_index = []
        for i in range(len(medoids)):
            medoid_index.append(int(med_input[medoids[i],3]))
        medoid_out = kg_output[medoid_index,:]
        logger.info("Clustering of Acquisition Function Evaluations Completed")
        return medoid_out       
    
    def __start_subprocesses__(self, subprocess_count):
        try:
            os.mkdir('{}/subprocess'.format(self.workingDir))
            logger.debug("Subprocess Directory Created")
        except FileExistsError:
            logger.debug("Subprocess Directory Already Exists")
            pass
        try:
            os.mkdir('{}/subprocess/LSFOut'.format(self.workingDir))
            logger.debug("LSFOut Directory Created")
        except FileExistsError:
            logger.debug("LSFOut Directory Already Exists")
            pass
        # This string is used to create the job files for the subprocesses used when calculating the knowledge gradient
        with open("{}/data/processStrings".format(self.workingDir), 'rb') as f:
            processStrings = load(f)
        
        logger.info("Strings for Subprocess Shell Files Loaded")
        
        subProcessStr = processStrings[0]
        runProcessStr = processStrings[1]
        calculation_count = self.sampleCount*self.hpCount*(len(self.ROM))
        if calculation_count % subprocess_count == 0:
            calcPerProcess = int(calculation_count/subprocess_count)
        else:
            calcPerProcess = int(calculation_count/subprocess_count) + 1
        
        logger.info("{} Subprocess Jobs | {} Calculations per Subprocess".format(subprocess_count, calcPerProcess))
        # Start all subprocesses
        for fname in range(subprocess_count):
            with open("{}/subprocess/{}.sh".format(self.workingDir, fname), 'w') as f:
                f.write(subProcessStr.format(fname))
            with open("{}/subprocess/submit{}.sh".format(self.workingDir, fname), 'w') as f:
                f.write(runProcessStr.format(fname))
            
            os.chmod("{}/subprocess/submit{}.sh".format(self.workingDir, fname), 0o775)
            subprocess.run(["{}/subprocess/submit{}.sh".format(self.workingDir, fname)], shell=True)
        # wait for all subprocesses to start
        all_pending = True
        logger.info("Waiting for Subprocess Jobs to start")
        count = 0
        all_started = False
        while all_pending:
            sleep(30)
            total_started = 0
            for fname in range(subprocess_count):
                if os.path.exists("{}/subprocess/sub{}.start".format(self.workingDir, fname)):
                    total_started += 1
            count += 1
            if total_started == subprocess_count:
                all_pending = False
                all_started = True
                logger.info("All Subprocess Jobs Started Successfully")
            # waiting for 2 hours for all the subprocesses to start will stop the waiting
            # and return false from this function to say that all the processes weren't
            # started yet
            if count == 240:
                all_pending = False
                logger.critical("Subprocess Jobs Outstanding after 2 Hours | {}/{} Jobs Started".format(total_started, subprocess_count))
                
        return calcPerProcess, all_started
    
    def run_multi_node(self, nodes=5):
        logger.info("Start Multinode Calculation")
        calcPerProcess, all_started = self.__start_subprocesses__(nodes)
        if all_started:
            start_process = True
            while start_process:
                self.timeCheck = time()
                
                if self.keepSubRunning:
                    pass
                else:
                    for kk in range(nodes):
                        try:
                            os.remove("{}/subprocess/close{}".format(self.workingDir, kk))
                            os.remove("{}/subprocess/sub{}.control".format(self.workingDir, kk))
                            os.remove("{}/subprocess/sub{}.start".format(self.workingDir, kk))
                            logger.debug("Close File {} removed".format(kk))
                        except FileExistsError:
                            logger.debug("Close File {} does not exist".format(kk))
                            
                    calcPerProcess, all_started = self.__start_subprocesses__(nodes)
                    subProcessWait = True
                    while subProcessWait:
                        if all_started:
                            subProcessWait = False
                        else:
                            total_started = 0
                            for fname in range(nodes):
                                if os.path.exists("{}/subprocess/sub{}.start".format(self.workingDir, fname)):
                                    total_started += 1
                            if total_started == nodes:
                                all_started = True
                                logger.info("All Subprocess Jobs Started Successfully")
                    
                
                x_test, check = apply_constraints(self.sampleCount, 
                                              self.nDim, self.res,
                                              self.A, self.b, self.Aeq, self.beq, 
                                              self.lb, self.ub, self.constr_func)
                if check:
                    logger.debug("Multi Node - All constraints applied successfully")
                else:
                    logger.critical("Multi Node - Some or All Constraints Could Not Be Applied! Continuing Without Constraints")
                
                               
                new_mean = []
                # obtain predictions from the low-order GPs
                for iii in range(len(self.ROM)):
                    new, var = self.reificationObj.predict_low_order(x_test, iii)
                    new_mean.append(new)
                                            
                kg_output = [] 
                
                # Calculate the Knowledge Gradient for each of the test points in each
                # model for each set of hyperparameters
                
                parameters = []
                sub_fnames = []
                
                count = 0
                sub_count = 0
                                
                # Initialise the parameters and shell files for running the subprocess
                # calculations. Run each subprocess as it is created
                logger.info("Set Up Parameters for Acquisition Function Evaluation and submit to Subprocesses")
                for jj in range(len(self.ROM)):
                    for kk in range(self.sampleCount):
                        model_temp = deepcopy(self.reificationObj)
                        model_temp.update_GP(np.expand_dims(x_test[kk], axis=0), 
                                              np.expand_dims(np.array([new_mean[jj][kk]]), 
                                                        axis=0), jj)
        
                        for mm in range(self.hpCount):
                            parameters.append((1, model_temp, self.xFused, self.fusedModelHP[mm,:],
                                            self.covFunc, x_test, jj, kk, mm, self.sampleCount,
                                            self.modelParam['costs'], self.maxTM))
                            count += 1
                            if count == calcPerProcess:
                                fname = "{}".format(sub_count)
                                sub_fnames.append(fname)
                                
                                with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                                    control_param = [0, "iteration", self.acquisitionFunc]
                                    dump(control_param, f)
        
                                with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                                    dump(parameters, f)
                                
                                parameters = []
                                count = 0
                                sub_count += 1
                
                
                if parameters != []:
                    fname = "{}".format(sub_count)
                    sub_fnames.append(fname)
                    
                    with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                        control_param = [0, "iteration", self.acquisitionFunc]
                        dump(control_param, f)
        
                    with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                        dump(parameters, f)
                
                logger.info("Start Waiting for Results to Complete")
                calc_start = time()
                sleep(60)
                
                finished = 0
                
                process_costs = np.zeros((len(sub_fnames)))
                
                while finished < len(sub_fnames):
                    finished = 0
                    proc_count = 0
                    for sub_name in sub_fnames:
                        with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_name), 'rb') as f:
                            control_param = load(f)
                            if control_param[0] == 1:
                                finished += 1
                                if process_costs[proc_count] == 0:
                                    process_costs[proc_count] = time()-calc_start
                    if finished < len(sub_fnames):          
                        sleep(60)
                
                logger.info("Acquisition Function Evaluations Completed")
                print("KG Results are Ready")
                process_cost = np.sum(process_costs)
                
                kg_output = []
                for sub_name in sub_fnames:
                    cont_loop = True
                    load_failed = True
                    timer = 0
                    while cont_loop:
                        try:
                            with open("{}/subprocess/{}.output".format(self.workingDir, sub_name), 'rb') as f:
                                sub_output = load(f)
                            load_failed = False
                            cont_loop = False
                        except FileNotFoundError:
                            sleep(30)
                            timer += 30
                        if timer > 300:
                            cont_loop = False
                            
                    if not load_failed:
                        for jj in range(len(sub_output)):
                            kg_output.append(sub_output[jj])
                        os.remove("{}/subprocess/{}.output".format(self.workingDir, sub_name))
                        os.remove("{}/subprocess/{}.dump".format(self.workingDir, sub_name))
                logger.debug("Calculation Results retrieved from Subprocess Jobs")
                # convert to a numpy array for ease of indexing
                kg_output = np.array(kg_output, dtype=object)
                
                medoid_out = self.__kg_calc_clustering(kg_output)
                                
                model_cost = time()-self.timeCheck + process_cost
                
                self.timeCheck = time()
                
                params = []
                count = np.zeros((len(self.ROM)+1)) 
                current = np.array(self.iterationData.iloc[:,3:])[-1,:]
                count[0:len(self.ROM)] = current[1:]
                count[-1] = current[0]
                param_index = 0
                logger.debug("Define Parameters for ROM Function Evaluations")
                for iii in range(medoid_out.shape[0]):
                    x_index = 6 + self.nDim
                    params.append({"Model Index":medoid_out[iii,3],
                                   "Model":self.ROM[medoid_out[iii,3]],
                                   "Input Values":np.array(medoid_out[iii,x_index:], dtype=np.float),
                                   "ParamIndex":param_index})
                    param_index += 1
    
                temp_x = np.zeros((len(params), self.nDim))
                temp_y = np.zeros(len(params))
                temp_index = np.zeros(len(params)) 
                costs = np.zeros(len(params))
                logger.info("Start ROM Function Evaluations | {} Calculations".format(len(params)))
                with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                    for result_from_process in zip(params, executor.map(call_model, params)):
                        par, results = result_from_process
                        self.reificationObj.update_GP(par["Input Values"], results, par["Model Index"])
                        temp_x[par["ParamIndex"],:] = par["Input Values"]
                        temp_y[par["ParamIndex"]] = results
                        temp_index[par["ParamIndex"]] = par["Model Index"]
                        costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                        count[par["Model Index"]] += 1
                self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
                self.totalBudgetLeft -= np.sum(costs) + model_cost
                self.tmBudgetLeft -= np.sum(costs) + model_cost
                logger.info("ROM Function Evaluations Completed")
                
                if (self.tmBudgetLeft < 0) or (self.tmIterCount == self.tmIterLim):
                    logger.info("Start Truth Model Evaluations")
                    self.tmIterCount = 0
                    self.tmBudgetLeft = self.tmBudget
                    
                    # create a test set that is dependent on the number of dimensions            
                    tm_test = lhs(self.nDim, samples=2500*self.nDim)
                    tm_test, check = apply_constraints(2500*self.nDim, 
                                              self.nDim, self.res,
                                              self.A, self.b, self.Aeq, self.beq, 
                                              self.lb, self.ub, self.constr_func)
                    if check:
                        logger.debug("Multi Node - All constraints applied successfully")
                    else:
                        logger.critical("Multi Node - Some or All Constraints Could Not Be Applied! Continuing Without Constraints")
                    
                    parameters = []
                    
                    # initialize the parameters for the fused model calculations and
                    # start the calculation
                    logger.debug("Define Parameters for Max Value Evaluations")
                    for mm in range(self.hpCount):
                        parameters.append((1, self.reificationObj, self.xFused, self.fusedModelHP[mm,:],
                                        self.covFunc, tm_test, self.maxTM, 0.01))
                        
                    with open("{}/subprocess/fused.dump".format(self.workingDir), 'wb') as f:
                        dump(parameters, f)
                        
                    with open("{}/subprocess/sub0.control".format(self.workingDir), 'wb') as f:
                        control_param = [0, "fused", self.acquisitionFunc]
                        dump(control_param, f)
                    logger.info("Parameters for Max Value Calculations Sent to Subprocess")
                    sleep(60)

                    while True:
                        if os.path.exists("{}/subprocess/fused.output".format(self.workingDir)):
                            break
                        else:
                            sleep(30)
                            
                    with open("{}/subprocess/fused.output".format(self.workingDir), 'rb') as f:
                        fused_output = load(f)
                        
                    os.remove("{}/subprocess/fused.dump".format(self.workingDir))
                    os.remove("{}/subprocess/fused.output".format(self.workingDir))

                    logger.info("Max Value Calculations Completed")
                    print("MAX Fused Calculations Completed")
                    # cluster the output from the fused model calculations
                    fused_output = np.array(fused_output, dtype=object)
                    try:
                        medoids, clusters = k_medoids(fused_output, self.batchSize)
                    except:
                        if (self.nDim > 1) and (self.nDim < self.batchSize):
                            try:
                                medoids, clusters = k_medoids(fused_output, self.nDim)
                            except:
                                medoids, clusters = k_medoids(fused_output, 1)
                        else:
                            medoids, clusters = k_medoids(fused_output, 1)
                    # Calculate the new Truth Model values and add them to the data
                    params = []
                    count = np.zeros((len(self.ROM)+1)) 
                    current = np.array(self.iterationData.iloc[:,3:])[-1,:]
                    count[0:len(self.ROM)] = current[1:]
                    count[-1] = current[0]
                    param_index = 0
                    logger.debug("Define Parameters for Truth Model Evaluations")
                    for iii in range(len(medoids)):
                        x_index = 6 + self.nDim
                        params.append({"Model Index":-1,
                                       "Model":self.TM,
                                       "Input Values":np.array(tm_test[fused_output[medoids[iii],1],:], dtype=np.float),
                                       "ParamIndex":param_index})
                        param_index += 1
                    
                    if self.externalTM or not self.keepSubRunning:
                        self.external_TM_data_save(params, count)
                        for fname in sub_fnames:
                            with open("{}/subprocess/close{}".format(self.workingDir, fname), 'w') as f:
                                f.write("Close Subprocess {}".format(fname))
                        start_process = False
                        break
                    else:
                        temp_x = np.zeros((len(params), self.nDim))
                        temp_y = np.zeros(len(params))
                        temp_index = np.zeros(len(params)) 
                        costs = np.zeros(len(params))
                        logger.info("Start Truth Model Evaluations | {} Sets".format(len(params)))
                        with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                            for result_from_process in zip(params, executor.map(call_model, params)):
                                par, results = result_from_process
                                self.reificationObj.update_truth(par["Input Values"], results)
                                temp_x[par["ParamIndex"],:] = par["Input Values"]
                                temp_y[par["ParamIndex"]] = results
                                temp_index[par["ParamIndex"]] = par["Model Index"]
                                costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                                count[par["Model Index"]] += 1
                        logger.info("Truth Model Evaluations Completed")
                        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
                        self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
                        if np.max(temp_y) > self.maxTM:
                            self.maxTM = np.max(temp_y)
                    
                self.__add_to_iterationData(time()-self.timeCheck, count)
                self.timeCheck = time()
                
                if self.updateROMafterTM:
                    self.__update_reduced_order_models__()
        
                self.__save_output_dataframes__()
                self.__save_calculation_state__()
                logger.info("Iteration {} Completed Successfully".format(self.currentIteration))

                if (self.totalBudgetLeft < 0) or (self.currentIteration >= self.iterLimit):
                    logger.info("Iteration or Budget Limit Met or Exceeded | Multi-Node Calculation Completed")
                    start_process = False
                
                self.currentIteration += 1
                self.tmIterCount += 1
    
    def __update_reduced_order_models__(self):
        logger.info("Recalculate all evaluated points for ROM to ensure correct model results are used")
        self.ROMInitInput = []
        self.ROMInitOutput = []
        TMDataX = self.reificationObj.x_true
        TMDataY = self.reificationObj.y_true
        params = []
        count = []
        param_index = 0
        for ii in range(len(self.ROM)):
            count.append(0)
            for jj in range(self.initDataPathorNum[ii]):
                params.append({"Model Index":ii,
                               "Model":self.ROM[ii],
                               "Input Values":self.reificationObj.x_train[ii][jj,:],
                               "ParamIndex":param_index})
                param_index += 1
            self.ROMInitInput.append(np.zeros_like(self.reificationObj.x_train[ii]))
            self.ROMInitOutput.append(np.zeros_like(self.reificationObj.y_train[ii]))
            
        temp_x = np.zeros((len(params), self.nDim))
        temp_y = np.zeros(len(params))
        temp_index = np.zeros(len(params))
                    
        with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
            for result_from_process in zip(params, executor.map(call_model, params)):
                par, results = result_from_process
                if par["Model Index"] != -1:
                    self.ROMInitInput[par["Model Index"]][count[par["Model Index"]],:] = par["Input Values"]
                    self.ROMInitOutput[par["Model Index"]][count[par["Model Index"]]] = results
                    temp_x[par["ParamIndex"],:] = par["Input Values"]
                    temp_y[par["ParamIndex"]] = results
                    temp_index[par["ParamIndex"]] = par["Model Index"]
        logger.info("Create New Reification Object")
        self.reificationObj = model_reification(self.ROMInitInput, self.ROMInitOutput, 
                                          self.modelParam['model_l'], 
                                          self.modelParam['model_sf'], 
                                          self.modelParam['model_sn'], 
                                          self.modelParam['means'], 
                                          self.modelParam['std'], 
                                          self.modelParam['err_l'], 
                                          self.modelParam['err_sf'], 
                                          self.modelParam['err_sn'], 
                                          TMDataX, TMDataY, 
                                          len(self.ROM), self.nDim, self.covFunc)
        
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.__add_to_iterationData(time()-self.timeCheck, np.array(count))
        self.timeCheck = time()
        logger.info("New Evaluations Saved | Reification Object Updated")
        pass
    
    def external_TM_data_save(self, TMEvaluationPoints, count):
        outputData = np.zeros((len(TMEvaluationPoints), self.nDim+1))
        for ii in range(len(TMEvaluationPoints)):
            outputData[ii,0:self.nDim] = TMEvaluationPoints[ii]["Input Values"]
            
        colNames = self.inputLabels.append("y")
        outputData = pd.DataFrame(outputData, columns=colNames)
        outputData.to_csv('{}/results/{}/TruthModelEvaluationPoints.csv'.format(self.workingDir, 
                                                                                self.calculationName))
        with open('{}/results/{}/countData'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(count, f)
        self.__save_calculation_state__()
        logger.critical("Truth Model Evaluation Points Copied to File | Restart Process when results are ready")
    
    def external_TM_data_load(self):
        self.__load_from_save__()
        with open('{}/results/{}/countData'.format(self.workingDir, self.calculationName), 'rb') as f:
            count = load(f)
        TMData = pd.read_csv('{}/results/{}/TruthModelEvaluationPoints.csv'.format(self.workingDir, 
                                                                                             self.calculationName))
        TMData = np.array(TMData)
        
        temp_x = np.zeros((TMData.shape[0], self.nDim))
        temp_y = np.zeros((TMData.shape[0]))
        temp_index = np.zeros((TMData.shape[0]))
        
        for ii in range(TMData.shape[0]):
            temp_x[ii,:] = TMData[ii,0:self.nDim]
            temp_y[ii] = TMData[ii,self.nDim+1]
            temp_index[ii] = -1
            count[-1] += 1
        
        logger.info("Truth Model Evaluations Loaded")
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
        
        if np.max(temp_y) > self.maxTM:
            self.maxTM = np.max(temp_y)
            
        self.__add_to_iterationData(time()-self.timeCheck, count)
        self.timeCheck = time()
        
        if self.updateROMafterTM:
            self.__update_reduced_order_models__()

        self.__save_output_dataframes__()
        self.__save_calculation_state__()
        logger.info("Iteration {} Completed Successfully".format(self.currentIteration))
        self.currentIteration += 1
        self.tmIterCount += 1
        
        
        
def runExternalTMCode(setupParams, InitialParams):
    framework = barefoot(**setupParams)
    if os.path.exists('{}/results/{}/TruthModelEvaluationPoints.csv'.format(framework.workingDir, framework.calculationName)):
        framework.external_TM_data_load()
        framework.run_single_node()
    else:
        framework.initialize_parameters(**InitialParams)
        framework.run_single_node()
    
    



##############################################################################
##############################################################################
##                                                                          ##
##                        Test Code Section                                 ##
##                                                                          ##
##############################################################################
##############################################################################

import math
import matplotlib.pyplot as plt

def rom1(x):
    a = 4.3
    return np.sin(a) + ((-np.cos(a))/math.factorial(1))*(x-a) + ((-np.sin(a))/math.factorial(2))*((x-a)**2) + ((np.cos(a))/math.factorial(3))*((x-a)**3) + ((np.sin(a))/math.factorial(4))*((x-a)**4) + ((-np.cos(a))/math.factorial(5))*((x-a)**5) + ((-np.sin(a))/math.factorial(6))*((x-a)**6) + ((np.cos(a))/math.factorial(7))*((x-a)**7)

def rom2(x):
    a = 4.5
    return np.sin(a) + ((-np.cos(a))/math.factorial(1))*(x-a) + ((-np.sin(a))/math.factorial(2))*((x-a)**2) + ((np.cos(a))/math.factorial(3))*((x-a)**3) + ((np.sin(a))/math.factorial(4))*((x-a)**4) + ((-np.cos(a))/math.factorial(5))*((x-a)**5) + ((-np.sin(a))/math.factorial(6))*((x-a)**6) + ((np.cos(a))/math.factorial(7))*((x-a)**7)

def rom3(x):
    a = 4.9
    return np.sin(a) + ((-np.cos(a))/math.factorial(1))*(x-a) + ((-np.sin(a))/math.factorial(2))*((x-a)**2) + ((np.cos(a))/math.factorial(3))*((x-a)**3) + ((np.sin(a))/math.factorial(4))*((x-a)**4) + ((-np.cos(a))/math.factorial(5))*((x-a)**5) + ((-np.sin(a))/math.factorial(6))*((x-a)**6) + ((np.cos(a))/math.factorial(7))*((x-a)**7)

def tm(x):
    return np.sin(x)





def plot_results(calcName):
    x = np.linspace(4,8,1000)

    y1 = tm(x)
    y2 = rom1(x)
    y3 = rom2(x)
    y4 = rom3(x)
    
    
    plt.figure()
    plt.plot(x,y1,label="TM")
    plt.plot(x,y2,label="ROM1")
    plt.plot(x,y3,label="ROM2")
    plt.plot(x,y4,label="ROM3")
    plt.legend()

    with open('./results/{}/iterationData'.format(calcName), 'rb') as f:
        iterationData = load(f)
    
    plt.figure()
    plt.plot(iterationData.loc[:,"Iteration"], iterationData.loc[:,"Max Found"])





def singeNodeTest():
    np.random.seed(100)
    ROMList = [rom1, rom2, rom3]
    test = barefoot(ROMModelList=ROMList, TruthModel=tm, 
                    calcInitData=True, initDataPathorNum=[1,1,1,1], nDim=1, 
                    calculationName="SingleNodeTest", acquisitionFunc="EI")
    modelParam = {'model_l':[[0.1],[0.1],[0.1]], 
                'model_sf':[1,1,1], 
                'model_sn':[0.01,0.01,0.01], 
                'means':[0,0,0], 
                'std':[1,1,1], 
                'err_l':[[0.1],[0.1],[0.1]], 
                'err_sf':[1,1,1], 
                'err_sn':[0.01,0.01,0.01],
                'costs':[1,2,3,20]}
    test.initialize_parameters(modelParam=modelParam, iterLimit=30, 
                               sampleCount=10, hpCount=50, 
                               batchSize=2, tmIter=5)
    test.run_single_node()
    
    plot_results("SingleNodeTest")
    
    
def externalTMSingeNodeTest():
    print("hello")
    num_dimensions = 1
    
    np.random.seed(100)
    ROMList = [rom1, rom2, rom3]
    test = barefoot(ROMModelList=ROMList, TruthModel=tm, 
                    calcInitData=True, initDataPathorNum=[1,1,1,1], nDim=num_dimensions, 
                    calculationName="SingleNodeExternalTMTest", acquisitionFunc="EI", 
                    externalTM=True)
    modelParam = {'model_l':[[0.1],[0.1],[0.1]], 
                'model_sf':[1,1,1], 
                'model_sn':[0.01,0.01,0.01], 
                'means':[0,0,0], 
                'std':[1,1,1], 
                'err_l':[[0.1],[0.1],[0.1]], 
                'err_sf':[1,1,1], 
                'err_sn':[0.01,0.01,0.01],
                'costs':[1,2,3,20]}
    test.initialize_parameters(modelParam=modelParam, iterLimit=30, 
                               sampleCount=10, hpCount=50, 
                               batchSize=2, tmIter=5)
    
    for ii in range(6):
        test.run_single_node()
        
        outputData = pd.read_csv('./results/SingleNodeExternalTMTest/TruthModelEvaluationPoints.csv',index_col=0)
        outputData = np.array(outputData)
        print(outputData)
        print(outputData[:,0:num_dimensions])
        
        values = tm(outputData[:,0:num_dimensions])
        
        outputData[:,num_dimensions] = values
        outputData = pd.DataFrame(outputData)
        print(outputData)
        outputData.to_csv('./results/SingleNodeExternalTMTest/TruthModelEvaluationPoints.csv')
        
        test.external_TM_data_load()        
    
    plot_results("SingleNodeExternalTMTest")
    

def multinode_Process_start(param):
    if param[0] == 2:
        param[1].run_multi_node(2)
        print("Main Code Finished")
        for fname in range(2):
            with open("subprocess/close{}".format(fname), 'w') as f:
                f.write("Close Subprocess {}".format(fname))
                print("Close Subprocess {}".format(fname))

        return "MultiNode Test Finished"
    else:
        subprocess.call(["python","subProcess.py","{}".format(param[0]),param[1]])
        return "Subprocess {} Finished".format(param[0])

def multiNodeTest():
    processStrings = ["this is a dummy file", "this is also a dummy file"]
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    with open('data/processStrings', 'wb') as f:
        dump(processStrings, f)

    ROMList = [rom1, rom2, rom3]
    
    framework = barefoot(ROMModelList=ROMList, TruthModel=tm, 
                    calcInitData=True, initDataPathorNum=[1,1,1,1], nDim=1, 
                    calculationName="MultiNodeTest", acquisitionFunc="EI")
    
    
    modelParam = {'model_l':[[0.1],[0.1],[0.1]], 
                'model_sf':[1,1,1], 
                'model_sn':[0.01,0.01,0.01], 
                'means':[0,0,0], 
                'std':[1,1,1], 
                'err_l':[[0.1],[0.1],[0.1]], 
                'err_sf':[1,1,1], 
                'err_sn':[0.01,0.01,0.01],
                'costs':[1,2,3,20]}
    
    framework.initialize_parameters(modelParam=modelParam, iterLimit=4, 
                                    sampleCount=3, hpCount=10, 
                                    batchSize=2, tmIter=2)
    
    params = [[0,framework.acquisitionFunc],[1,framework.acquisitionFunc],[2,framework]]
    
    with concurrent.futures.ProcessPoolExecutor(3) as executor:
        for result_from_process in zip(params, executor.map(multinode_Process_start, params)):
            print(result_from_process)
            
    plot_results("MultiNodeTest")
    
    

def extmultinode_Process_start(param):
    if param[0] == 2:
        if os.path.exists('./results/ExternalTMMultiNodeTest/TruthModelEvaluationPoints.csv'):
            param[1].external_TM_data_load()
        param[1].run_multi_node(2)
        print("Main Code Finished")
        for fname in range(2):
            with open("subprocess/close{}".format(fname), 'w') as f:
                f.write("Close Subprocess {}".format(fname))
                print("Close Subprocess {}".format(fname))

        return "MultiNode Test Finished"
    else:
        subprocess.call(["python","subProcess.py","{}".format(param[0]),param[1]])
        return "Subprocess {} Finished".format(param[0])    


def externalTMMultiNodeTest():
    processStrings = ["this is a dummy file", "this is also a dummy file"]
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    with open('data/processStrings', 'wb') as f:
        dump(processStrings, f)

    ROMList = [rom1, rom2, rom3]
    
    framework = barefoot(ROMModelList=ROMList, TruthModel=tm, 
                    calcInitData=True, initDataPathorNum=[1,1,1,1], nDim=1, 
                    calculationName="ExternalTMMultiNodeTest", acquisitionFunc="EI", 
                    externalTM=True)
    
    
    modelParam = {'model_l':[[0.1],[0.1],[0.1]], 
                'model_sf':[1,1,1], 
                'model_sn':[0.01,0.01,0.01], 
                'means':[0,0,0], 
                'std':[1,1,1], 
                'err_l':[[0.1],[0.1],[0.1]], 
                'err_sf':[1,1,1], 
                'err_sn':[0.01,0.01,0.01],
                'costs':[1,2,3,20]}
    
    framework.initialize_parameters(modelParam=modelParam, iterLimit=4, 
                                    sampleCount=3, hpCount=10, 
                                    batchSize=2, tmIter=2)
    
    params = [[0,framework.acquisitionFunc],[1,framework.acquisitionFunc],[2,framework]]
    
    for ii in range(2):
    
        with concurrent.futures.ProcessPoolExecutor(3) as executor:
            for result_from_process in zip(params, executor.map(multinode_Process_start, params)):
                print(result_from_process)
                
        outputData = pd.read_csv('./results/ExternalTMMultiNodeTest/TruthModelEvaluationPoints.csv',index_col=0)
        outputData = np.array(outputData)
        print(outputData)
        print(outputData[:,0:1])
        
        values = tm(outputData[:,0:1])
        
        outputData[:,1] = values
        outputData = pd.DataFrame(outputData)
        print(outputData)
        outputData.to_csv('./results/ExternalTMMultiNodeTest/TruthModelEvaluationPoints.csv')
        
        
        
        for iii in range(2):
            os.remove("subprocess/close{}".format(iii))
            
        
            
    plot_results("ExternalTMMultiNodeTest")



if __name__ == "__main__":
    # singeNodeTest()
    # multiNodeTest()
    # externalTMSingeNodeTest()
    externalTMMultiNodeTest()
    