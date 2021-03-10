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
import concurrent.futures
from multiprocessing import cpu_count
from copy import deepcopy
from util import k_medoids, cartesian, call_model, apply_constraints, calculate_KG, calculate_EI, fused_calculate, calculate_TS
import logging

# from testFrameworkCodeUtil import ThreeHumpCamel, ThreeHumpCamel_LO1, ThreeHumpCamel_LO2, ThreeHumpCamel_LO3, plotResults 
# from testFrameworkCodeUtil import isostress_IS, isostrain_IS, isowork_IS, EC_Mart_IS, secant1_IS, TC_GP, RVE_GP

class barefoot():
    def __init__(self, ROMModelList=[], TruthModel=[], calcInitData=True, 
                 initDataPathorNum=[], multiNode=0, workingDir=".", 
                 calculationName="Calculation", nDim=1, input_resolution=5, restore_calc=False,
                 updateROMafterTM=False, externalTM=False, acquisitionFunc="KG",
                 A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], keepSubRunning=True, 
                 verbose=False, sampleScheme="LHS", logname="BAREFOOT"):
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
        keepSubRunning     : Determines whether the subprocesses are left running while calling the Truth Model
        verbose            : Determines the logging level for tracking the calculations.
                             
        """
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        
        # create logger to output framework progress
        self.logger = logging.getLogger(logname)
        self.logger.setLevel(log_level)
        fh = logging.FileHandler('{}.log'.format(logname))
        fh.setLevel(log_level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add the handler to the logger
        self.logger.addHandler(fh)
        
        
        self.logger.info("#########################################################")
        self.logger.info("#                                                       #")
        self.logger.info("#        Start BAREFOOT Framework Initialization        #")
        self.logger.info("#                                                       #")
        self.logger.info("#########################################################")
        
        # Restore a previous calculation and restart the timer or load new
        # information and initialize
        
        if restore_calc:
            self.__load_from_save(workingDir, calculationName)
            self.timeCheck = time()
            self.logger.info("Previous Save State Restored")
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
            self.constr_func = func
            self.sampleScheme = sampleScheme
            self.keepSubRunning = keepSubRunning
            self.updateROMafterTM = updateROMafterTM
            self.externalTM = externalTM
            self.acquisitionFunc = acquisitionFunc
            self.__create_dir_and_files()
            self.__create_output_dataframes()
            self.__get_initial_data__()
            self.logger.info("Initialization Completed")  
        
    
    def __create_dir_and_files(self):
        # Create the required directories for saving the results and the subprocess
        # information if applicable
        try:
            os.mkdir('{}/results'.format(self.workingDir))
            self.logger.debug("Results Directory Created Successfully")
        except FileExistsError:
            self.logger.debug("Results Directory Already Exists")
        try:
            os.mkdir('{}/data'.format(self.workingDir))
            self.logger.debug("Data Directory Created Successfully")
        except FileExistsError:
            self.logger.debug("Data Directory Already Exists")
        try:
            os.mkdir('{}/data/parameterSets'.format(self.workingDir))
            self.logger.debug("Parameter Set Directory Created Successfully")
        except FileExistsError:
            self.logger.debug("Parameter Set Directory Already Exists")
        try:
            os.mkdir('{}/results/{}'.format(self.workingDir, 
                                            self.calculationName))
            self.logger.debug("Calculation Results Directory [{}] Created Successfully".format(self.calculationName))
        except FileExistsError:
            self.logger.debug("Calculation Results Directory [{}] Already Exists".format(self.calculationName))
        # If using subprocesses, create the folder structure needed
        if self.multinode != 0:
            if os.path.exists('{}/subprocess'.format(self.workingDir)):
                rmtree('{}/subprocess'.format(self.workingDir))
                self.logger.debug("Existing Subprocess Directory Removed")
            os.mkdir('{}/subprocess'.format(self.workingDir))
            os.mkdir('{}/subprocess/LSFOut'.format(self.workingDir))
            self.logger.debug("Subprocess Directory Created")
                
    def __create_output_dataframes(self):
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
        self.logger.debug("Output Dataframes Created")
    
    def __save_output_dataframes(self):
        # The dataframes are saved in two forms, first a pickled version of the
        # dataframe, and also a csv version for readability
        with open('{}/results/{}/evaluatedPoints'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(self.evaluatedPoints, f)
        self.evaluatedPoints.to_csv('{}/results/{}/evaluatedPoints.csv'.format(self.workingDir, self.calculationName))
        with open('{}/results/{}/iterationData'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(self.iterationData, f)
        self.iterationData.to_csv('{}/results/{}/iterationData.csv'.format(self.workingDir, self.calculationName))
        self.logger.info("Dataframes Pickled and Dumped to Results Directory")
            
    def __save_calculation_state(self):
        # This function saves the entire barefoot object into a pickle file
        with open('{}/data/{}_save_state'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(self, f)
        self.logger.info("Calculation State Saved")
        
    def __load_from_save(self, workingDir, calculationName):
        # This function restores the barefoot object parameters from a saved
        # pickle file. In order for this to work, each variable of the object
        # is restored separately.
        try:
            with open('{}/data/{}_save_state'.format(workingDir, calculationName), 'rb') as f:
                saveState = load(f)
                self.logger.debug("Save State File Found")
            for item in vars(saveState).items():
                setattr(self, item[0], item[1])
        except FileNotFoundError:
            self.loadFailed = True
            self.logger.warning("Could not find Save State File")
        
    def __add_to_evaluatedPoints(self, modelIndex, eval_x, eval_y):
        # Adds new data points to the evaluated datapoints dataframe
        temp = np.zeros((eval_x.shape[0], self.nDim+3))
        temp[:,0] = modelIndex
        temp[:,1] = self.currentIteration
        temp[:,2] = eval_y
        temp[:,3:] = eval_x
        temp = pd.DataFrame(temp, columns=self.evaluatedPoints.columns)
        self.evaluatedPoints = pd.concat([self.evaluatedPoints,temp])
        self.logger.debug("{} New Points Added to Evaluated Points Dataframe".format(eval_x.shape[0]))
        
    def __add_to_iterationData(self, calcTime, iterData):
        # Adds new data points to the Iteration Data Dataframe
        temp = np.zeros((1,4+len(self.ROM)))
        temp[0,0] = self.currentIteration
        temp[0,1] = self.maxTM
        temp[0,2] = calcTime
        temp[0,3] = iterData[-1]
        temp[0,4:] = iterData[0:len(self.ROM)]
        temp = pd.DataFrame(temp, columns=self.iterationData.columns)
        self.iterationData = pd.concat([self.iterationData,temp])
        self.logger.debug("Iteration {} Data saved to Dataframe".format(self.currentIteration))
        
    def __get_initial_data__(self):
        # Function for obtaining the initial data either by calculation or by 
        # extracting the data from a file.
        params = []
        count = []
        param_index = 0
        self.maxTM = -np.inf
        # Check if data needs to be calculated or extracted
        if self.calcInitData:
            self.logger.debug("Start Calculation of Initial Data")
            # obtain LHS initial data for each reduced order model
            for ii in range(len(self.ROM)):
                count.append(0)                
                initInput, check = apply_constraints(self.initDataPathorNum[ii], 
                                                     self.nDim, self.res,
                                                      self.A, self.b, self.Aeq, self.beq, 
                                                      self.lb, self.ub, self.constr_func)
                if check:
                    self.logger.debug("Initial Data - All constraints applied successfully")
                else:
                    self.logger.critical("Initial Data - Some or All Constraints Could not Be applied! Continuing Without Constraints")
                
                for jj in range(self.initDataPathorNum[ii]):
                    params.append({"Model Index":ii,
                                   "Model":self.ROM[ii],
                                   "Input Values":initInput[jj,:],
                                   "ParamIndex":param_index})
                    param_index += 1
                self.ROMInitInput.append(np.zeros_like(initInput))
                self.ROMInitOutput.append(np.zeros(self.initDataPathorNum[ii]))
            count.append(0)
            # Obtain LHS initial data for Truth Model
            initInput, check = apply_constraints(self.initDataPathorNum[ii+1], 
                                                     self.nDim, self.res,
                                                      self.A, self.b, self.Aeq, self.beq, 
                                                      self.lb, self.ub, self.constr_func)
            if check:
                self.logger.debug("Initial Data - All constraints applied successfully")
            else:
                self.logger.critical("Initial Data - Some or All Constraints Could not Be applied! Continuing Without Constraints")
            for jj in range(self.initDataPathorNum[-1]):
                params.append({"Model Index":-1,
                               "Model":self.TM,
                               "Input Values":initInput[jj,:],
                               "ParamIndex":param_index})
                param_index += 1
            self.TMInitInput = np.zeros_like(initInput)
            self.TMInitOutput = np.zeros(self.initDataPathorNum[-1])
            
            # Calculate all the initial data in parallel
            temp_x = np.zeros((len(params), self.nDim))
            temp_y = np.zeros(len(params))
            temp_index = np.zeros(len(params))
            self.logger.debug("Parameters Defined. Starting Concurrent.Futures Calculation")
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
                        if np.max(results) > self.maxTM:
                            self.maxTM = np.max(results)
                        temp_x[par["ParamIndex"],:] = par["Input Values"]
                        temp_y[par["ParamIndex"]] = results
                        temp_index[par["ParamIndex"]] = par["Model Index"]
                    count[par["Model Index"]] += 1
            self.logger.debug("Concurrent.Futures Calculation Completed")
        else:
            # extract the initial data from the file
            self.logger.debug("Start Loading Initial Data from Files")
            with open(self.initDataPathorNum, 'rb') as f:
                data = load(f)
            
            # extract data from dictionary in file and assign to correct variables
            self.TMInitOutput = data["TMInitOutput"]
            self.TMInitInput = data["TMInitInput"]
            self.ROMInitOutput = data["ROMInitOutput"]
            self.ROMInitInput = data["ROMInitInput"]
            
            ROMSize = 0
            for mmm in range(len(self.ROMInitInput)):
                ROMSize += self.ROMInitOutput[mmm].shape[0]
            
            temp_x = np.zeros((self.TMInitOutput.shape[0]+ROMSize, 
                               self.nDim))
            temp_y = np.zeros(self.TMInitOutput.shape[0]+ROMSize)
            temp_index = np.zeros(self.TMInitOutput.shape[0]+ROMSize)
            
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
                if self.TMInitOutput[jj] > self.maxTM:
                    self.maxTM = self.TMInitOutput[jj]
                temp_index[ind] = -1
                ind += 1
            count.append(self.TMInitInput.shape[0])
            self.logger.debug("Loading Data From File Completed")
        # Add initial data to dataframes
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.__add_to_iterationData(time()-self.timeCheck, np.array(count))
        self.logger.debug("Initial Data Saved to Dataframes")
        self.timeCheck = time()
    
    def initialize_parameters(self, modelParam, covFunc="M32", iterLimit=100,  
                              sampleCount=50, hpCount=100, batchSize=5, 
                              tmIter=1e6, totalBudget=1e16, tmBudget=1e16, 
                              upperBound=1, lowBound=0.0001, fusedPoints=5):
        """
        This function sets the conditions for the barefoot framework calculations.
        All parameters have default values except the model parameters.

        Parameters
        ----------
        modelParam : TYPE
            This must be a dictionary with the hyperparameters for the reduced
            order models as well as the costs for all the models. The specific
            values in the dictionary must be:
                'model_l': A list with the characteristic length scale for each
                           dimension in each reduced order model GP.
                           eg 2 reduced order - 3 dimension models
                           [[0.1,0.1,0.1],[0.2,0.2,0.2]]
                'model_sf': A list with the signal variance for each reduced
                            order model GP.
                'model_sn': A list with the noise variance for each reduced
                            order model GP.
                'means': A list of the mean of each model. Set to 0 if the mean
                         is not known
                'std': A list of the standard deviations of each model. Set to 1
                       if the standard deviation is not known.
                'err_l': A list with the characteristic length scale for each
                           dimension in each discrepancy GP. Must match dimensions
                           of model_l
                'err_sf': A list with the signal variance for each discrepancy GP.
                'err_sn': A list with the noise variance for each discrepancy GP.
                'costs': The model costs, including the Truth Model
                         eg. 2 ROM : [model 1 cost, model 2 cost, Truth model cost]
        covFunc : TYPE, optional
            The covariance function to used for the Gaussian Process models.
            Options are Squared Exponential ("SE") Matern 3/2 ("M32") and 
            Matern 5/2 ("M52"). The default is "M32".
        iterLimit : TYPE, optional
            How many iterations to run the framework calculation before
            terminating. The default is 100.
        sampleCount : TYPE, optional
            The number of samples to use for the acquisition function calculations.
            The default is 50.
        hpCount : TYPE, optional
            The number of hyperparameter sets to use. The default is 100.
        batchSize : TYPE, optional
            The batch size for the model evaluations. The default is 5.
        tmIter : TYPE, optional
            The number of iterations to complete before querying the Truth Model. 
            The default is 1e6.
        totalBudget : TYPE, optional
            The total time budget to expend before terminating the calculation. 
            The default is 1e16.
        tmBudget : TYPE, optional
            The budget to expend before querying the Truth Model. The default 
            is 1e16.
        upperBound : TYPE, optional
            The upper bound for the hyperparameters. The default is 1.
        lowBound : TYPE, optional
            The lower bound for the hyperparameters. The default is 0.0001.
        fusedPoints : TYPE, optional
            The number of points per dimension for the linear grid used to 
            evaluate the fused mean and variance for building the fused model. 
            The default is 5.

        Returns
        -------
        None.

        """
        self.logger.debug("Start Initializing Reification Object Parameters")
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
        self.logger.debug("Create Reification Object")
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
        self.logger.info("Reification Object Initialized. Ready for Calculations")
        
    def __restart_subs(self):
        for kk in range(self.multinode):
            try:
                os.remove("{}/subprocess/close{}".format(self.workingDir, kk))
                os.remove("{}/subprocess/sub{}.control".format(self.workingDir, kk))
                os.remove("{}/subprocess/sub{}.start".format(self.workingDir, kk))
                self.logger.debug("Close File {} removed".format(kk))
            except FileExistsError:
                self.logger.debug("Close File {} does not exist".format(kk))
                
        calcPerProcess, all_started = self.__start_subprocesses__(self.multinode)
        subProcessWait = True
        while subProcessWait:
            if all_started:
                subProcessWait = False
            else:
                total_started = 0
                for fname in range(self.multinode):
                    if os.path.exists("{}/subprocess/sub{}.start".format(self.workingDir, fname)):
                        total_started += 1
                if total_started == self.multinode:
                    all_started = True
                    self.logger.info("All Subprocess Jobs Started Successfully")
                    
    def __run_multinode_acq_func(self, x_test, new_mean, calcPerProcess):
        self.logger.info("Set Up Parameters for Acquisition Function Evaluation and submit to Subprocesses")
        parameters = []
        parameterFileData = []
        sub_fnames = []
        count = 0
        sub_count = 0
        parameterIndex = 0
        parameterFileIndex = 0
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
        for jj in range(len(self.ROM)):
            for kk in range(self.sampleCount):
                model_temp = [np.expand_dims(x_test[kk], axis=0), 
                              np.expand_dims(np.array([new_mean[jj][kk]]), axis=0), 
                              jj]

                for mm in range(self.hpCount):
                    parameterFileData.append((1, model_temp, self.xFused, self.fusedModelHP[mm,:],
                                    self.covFunc, x_test, jj, kk, mm, self.sampleCount,
                                    self.modelParam['costs'], self.maxTM))
                    parameters.append([parameterIndex, parameterFileIndex])
                    parameterIndex += 1
                    
                    if len(parameterFileData) == 1000:
                        with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                            dump(parameterFileData, f)
                        parameterFileData = []
                        parameterFileIndex += 1
                        parameterIndex = 0
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
        
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
        
        if parameters != []:
            fname = "{}".format(sub_count)
            sub_fnames.append(fname)
            
            with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                control_param = [0, "iteration", self.acquisitionFunc]
                dump(control_param, f)

            with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                dump(parameters, f)
                
        self.logger.info("Start Waiting for Results to Complete")
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
        
        self.logger.info("Acquisition Function Evaluations Completed")
        
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
                self.logger.debug("sub_output {} found | length: {}".format(sub_name, len(sub_output)))
                for jj in range(len(sub_output)):
                    kg_output.append(sub_output[jj])
                os.remove("{}/subprocess/{}.output".format(self.workingDir, sub_name))
                os.remove("{}/subprocess/{}.dump".format(self.workingDir, sub_name))
            else:
                self.logger.debug("sub_output {} NOT found".format(len(sub_name)))
        self.logger.debug("Calculation Results retrieved from Subprocess Jobs")
        return kg_output, process_cost
    
    def __run_singlenode_acq_func(self, x_test, new_mean):        
        parameters = []
        parameterFileData = []
        count = 0
        parameterIndex = 0
        parameterFileIndex = 0
        self.logger.debug("Set Up Parameters for Acquisition Function Evaluation")
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
        for jj in range(len(self.ROM)):
            for kk in range(self.sampleCount):
                
                
                # model_temp = deepcopy(self.reificationObj)
                model_temp = [np.expand_dims(x_test[kk], axis=0), 
                              np.expand_dims(np.array([new_mean[jj][kk]]), axis=0), 
                              jj]

                for mm in range(self.hpCount):
                    parameterFileData.append((1, model_temp, self.xFused, self.fusedModelHP[mm,:],
                                    self.covFunc, x_test, jj, kk, mm, self.sampleCount,
                                    self.modelParam['costs'], self.maxTM))
                    parameters.append([parameterIndex, parameterFileIndex])
                    parameterIndex += 1
                    
                    if len(parameterFileData) == 1000:
                        with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                            dump(parameterFileData, f)
                        parameterFileData = []
                        parameterFileIndex += 1
                        parameterIndex = 0
                    count += 1
                    
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
                    
        if self.acquisitionFunc == "EI":
            acqFunc = calculate_EI
        elif self.acquisitionFunc == "KG":
            acqFunc = calculate_KG
        elif self.acquisitionFunc == "TS":
            acqFunc = calculate_TS
        kg_output = []
        self.logger.info("Start Acquisition Function Evaluations for {} Parameter Sets".format(len(parameters)))
        with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
            for result_from_process in zip(parameters, executor.map(acqFunc,parameters)):
                params, results = result_from_process
                kg_output.append(results)
        self.logger.info("Acquisition Function Evaluations Completed")
        return kg_output, 0
    
    def __run_multinode_fused(self, tm_test):
        # initialize the parameters for the fused model calculations and
        # start the calculation
        calc_limit = (-(-self.hpCount//self.multinode)) 
        self.logger.debug("Define Parameters for Max Value Evaluations")
        parameters = []
        parameterFileData = []
        parameterIndex = 0
        parameterFileIndex = 0
        count = 0
        sub_count = 0
        sub_fnames = []
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
        for mm in range(self.hpCount):
            parameterFileData.append((1, [], self.xFused, self.fusedModelHP[mm,:],
                            self.covFunc, tm_test, self.maxTM, 0.01))
            parameters.append([parameterIndex, parameterFileIndex])
            parameterIndex += 1
            count += 1
            
            if len(parameterFileData) == 500:
                with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                    dump(parameterFileData, f)
                parameterFileData = []
                parameterFileIndex += 1
                parameterIndex = 0
            
            if count == calc_limit:
                fname = "{}".format(sub_count)
                sub_fnames.append(fname)
                
                with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                    control_param = [0, "fused", self.acquisitionFunc]
                    dump(control_param, f)

                with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                    dump(parameters, f)
                
                parameters = []
                count = 0
                sub_count += 1
                
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
                
        if parameters != []:
            fname = "{}".format(sub_count)
            sub_fnames.append(fname)
            
            with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                control_param = [0, "fused", self.acquisitionFunc]
                dump(control_param, f)

            with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                dump(parameters, f)

        self.logger.info("Parameters for Max Value Calculations Sent to Subprocess")
        sleep(60)

        finished = 0
        while finished < len(sub_fnames):
            finished = 0
            for sub_name in sub_fnames:
                with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_name), 'rb') as f:
                    control_param = load(f)
                    if control_param[0] == 1:
                        finished += 1
            if finished < len(sub_fnames):          
                sleep(60)        
        
        fused_output = []
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
                self.logger.debug("sub_output {} found | length: {}".format(sub_name, len(sub_output)))
                for jj in range(len(sub_output)):
                    fused_output.append(sub_output[jj])
                os.remove("{}/subprocess/{}.output".format(self.workingDir, sub_name))
                os.remove("{}/subprocess/{}.dump".format(self.workingDir, sub_name))
            else:
                self.logger.debug("sub_output {} NOT found".format(len(sub_name)))
        
        # with open("{}/subprocess/fused.output".format(self.workingDir), 'rb') as f:
        #     fused_output = load(f)
            
        # os.remove("{}/subprocess/fused.dump".format(self.workingDir))
        # os.remove("{}/subprocess/fused.output".format(self.workingDir))

        self.logger.info("Max Value Calculations Completed")
        return fused_output
        
    def __run_singlenode_fused(self, tm_test):
        parameters = []
        parameterFileData = []
        # initialize the parameters for the fused model calculations and
        # start the calculation
        self.logger.debug("Define Parameters for Max Value Evaluations")
        parameterIndex = 0
        parameterFileIndex = 0
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
        for mm in range(self.hpCount):
            parameterFileData.append((1, [], self.xFused, self.fusedModelHP[mm,:],
                            self.covFunc, tm_test, self.maxTM, 0.01))
            parameters.append([parameterIndex, parameterFileIndex])
            parameterIndex += 1
            if len(parameterFileData) == 500:
                with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                    dump(parameterFileData, f)
                parameterFileData = []
                parameterFileIndex += 1
                parameterIndex = 0
            
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)

        fused_output = []
        self.logger.info("Start Max Value Calculations | {} Sets".format(len(parameters)))
        count = 0
        with concurrent.futures.ProcessPoolExecutor(2) as executor:#cpu_count()) as executor:
            for result_from_process in zip(parameters, executor.map(fused_calculate,parameters)):
                params, results = result_from_process
                fused_output.append(results[0])
                count += 1
                
        max_values = np.zeros((results[1],2))
                    
        for ii in range(len(fused_output)):
            if max_values[fused_output[ii][1],0] != 0:
                if max_values[fused_output[ii][1],0] < fused_output[ii][0]:
                    max_values[fused_output[ii][1],0] = fused_output[ii][0]
                    max_values[fused_output[ii][1],1] = fused_output[ii][1]
            else:
                max_values[fused_output[ii][1],0] = fused_output[ii][0]
                max_values[fused_output[ii][1],1] = fused_output[ii][1]
                    
        fused_output = max_values[np.where(max_values[:,0]!=0)]
        self.logger.info("Max Value Calculations Completed")
        return fused_output

    def __call_ROM(self, medoid_out):
        params = []
        count = np.zeros((len(self.ROM)+1)) 
        current = np.array(self.iterationData.iloc[:,3:])[-1,:]
        count[0:len(self.ROM)] = current[1:]
        count[-1] = current[0]
        param_index = 0
        self.logger.debug("Define Parameters for ROM Function Evaluations")
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
        self.logger.info("Start ROM Function Evaluations | {} Calculations".format(len(params)))
        with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
            for result_from_process in zip(params, executor.map(call_model, params)):
                par, results = result_from_process
                self.reificationObj.update_GP(par["Input Values"], results, par["Model Index"])
                temp_x[par["ParamIndex"],:] = par["Input Values"]
                temp_y[par["ParamIndex"]] = results
                temp_index[par["ParamIndex"]] = par["Model Index"]
                costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                count[par["Model Index"]] += 1
        return temp_x, temp_y, temp_index, costs, count
    
    def __call_Truth(self, params, count):
        temp_x = np.zeros((len(params), self.nDim))
        temp_y = np.zeros(len(params))
        temp_index = np.zeros(len(params)) 
        costs = np.zeros(len(params))
        self.logger.info("Start Truth Model Evaluations | {} Sets".format(len(params)))
        with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
            for result_from_process in zip(params, executor.map(call_model, params)):
                par, results = result_from_process
                costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                if results != False:
                    self.reificationObj.update_truth(par["Input Values"], results)
                    temp_x[par["ParamIndex"],:] = par["Input Values"]
                    temp_y[par["ParamIndex"]] = results
                    temp_index[par["ParamIndex"]] = par["Model Index"]
                    count[par["Model Index"]] += 1
        temp_x = temp_x[np.where(temp_y != 0)]
        temp_y = temp_y[np.where(temp_y != 0)]
        temp_index = temp_index[np.where(temp_y != 0)]
        self.logger.info("Truth Model Evaluations Completed")
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
        if np.max(temp_y) > self.maxTM:
            self.maxTM = np.max(temp_y)
        return count
    
    def run_optimization(self):
        self.logger.info("Start BAREFOOT Framework Calculation")
        # Check if the calculation requires multiple nodes and start them if necessary
        if self.multinode > 0:
            calcPerProcess, all_started = self.__start_subprocesses__(self.multinode)
        else:
            calcPerProcess, all_started = (0, True)
        # Once all subprocesses have started, start the main calculation
        if all_started:
            start_process = True
            while start_process:
                text_num = str(self.currentIteration)
                self.logger.info("#########################################################")
                self.logger.info("#                Start Iteration : {}                 #".format("0"*(4-len(text_num))+text_num))
                self.logger.info("#########################################################")
                self.timeCheck = time()
                # for multinode calculations, check if subprocesses are being kept
                # running and restart if not
                if self.keepSubRunning:
                    pass
                else:
                    self.__restart_subs()
                # Check constraints and obtain latin-hypercube sampled test points
                x_test, check = apply_constraints(self.sampleCount, 
                                              self.nDim, self.res,
                                              self.A, self.b, self.Aeq, self.beq, 
                                              self.lb, self.ub, self.constr_func,
                                              self.sampleScheme)
                # If constraints can't be satisfied, notify the user in the log
                if check:
                    self.logger.debug("ROM - All constraints applied successfully {}/{}".format(x_test.shape[0], self.sampleCount))
                else:
                    self.logger.critical("ROM - Sample Size NOT met due to constraints! Continue with {}/{} Samples".format(x_test.shape[0], self.sampleCount))
                
                new_mean = []
                # obtain predictions from the low-order GPs
                for iii in range(len(self.ROM)):
                    new, var = self.reificationObj.predict_low_order(x_test, iii)
                    new_mean.append(new)
                
                # Calculate the Acquisition Function for each of the test points in each
                # model for each set of hyperparameters
                
                if self.multinode > 0:
                    kg_output, process_cost = self.__run_multinode_acq_func(x_test, 
                                                                          new_mean, 
                                                                          calcPerProcess)
                else:
                    kg_output, process_cost = self.__run_singlenode_acq_func(x_test, 
                                                             new_mean)
                
                kg_output = np.array(kg_output, dtype=object)
                
                # Cluster the acquisition function output
                medoid_out = self.__kg_calc_clustering(kg_output)
                
                model_cost = time()-self.timeCheck + process_cost
                self.timeCheck = time()
                
                # Call the reduced order models
                temp_x, temp_y, temp_index, costs, count = self.__call_ROM(medoid_out)
                
                self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
                self.totalBudgetLeft -= np.sum(costs) + model_cost
                self.tmBudgetLeft -= np.sum(costs) + model_cost
                self.logger.info("ROM Function Evaluations Completed")
                
                if (self.tmBudgetLeft < 0) or (self.tmIterCount == self.tmIterLim):
                    self.logger.info("Start Truth Model Evaluations")
                    
                    # create a test set that is dependent on the number of dimensions            
                    tm_test, check = apply_constraints(2500*self.nDim, 
                                              self.nDim, self.res,
                                              self.A, self.b, self.Aeq, self.beq, 
                                              self.lb, self.ub, self.constr_func, False)
                    if check:
                        self.logger.debug("Truth Model Query - All constraints applied successfully")
                    else:
                        self.logger.critical("Truth Model Query - Some or All Constraints Could Not Be Applied! Continuing Without Constraints")
                    
                    if self.multinode > 0:
                        fused_output = self.__run_multinode_fused(tm_test)
                    else:
                        fused_output = self.__run_singlenode_fused(tm_test)
                    
                    fused_output = np.array(fused_output, dtype=object)
                    if fused_output.shape[0] > self.batchSize:
                        medoids, clusters = k_medoids(fused_output, self.batchSize)
                    elif (self.nDim > 1) and (self.nDim < self.batchSize) and fused_output.shape[0] > self.nDim:
                        medoids, clusters = k_medoids(fused_output, self.nDim)
                    else:
                        medoids = []
                        for iii in range(fused_output.shape[0]):
                            medoids.append(iii)
                                            
                    params = []
                    param_index = 0
                    self.logger.debug("Define Parameters for Truth Model Evaluations")
                    for iii in range(len(medoids)):
                        params.append({"Model Index":-1,
                                       "Model":self.TM,
                                       "Input Values":np.array(tm_test[int(fused_output[medoids[iii],1]),:], dtype=np.float),
                                       "ParamIndex":param_index})
                        param_index += 1
                    
                    self.tmIterCount = 0
                    self.tmBudgetLeft = self.tmBudget
                    
                    if self.externalTM or not self.keepSubRunning:
                        self.__external_TM_data_save(params, count)
                        if self.multinode > 0:
                            for fname in range(self.multinode):
                                with open("{}/subprocess/close{}".format(self.workingDir, fname), 'w') as f:
                                    f.write("Close Subprocess {}".format(fname))
                            start_process = False
                        break
                    else:
                        count = self.__call_Truth(params, count)
                    
                self.__add_to_iterationData(time()-self.timeCheck + model_cost, count)
                self.timeCheck = time()
                
                if self.updateROMafterTM:
                    self.__update_reduced_order_models__()
        
                self.__save_output_dataframes()
                self.__save_calculation_state()
                self.logger.info("Iteration {} Completed Successfully".format(self.currentIteration))
    
                if (self.totalBudgetLeft < 0) or (self.currentIteration >= self.iterLimit):
                    self.logger.info("#########################################################")
                    self.logger.info("#                                                       #")
                    self.logger.info("#       Iteration or Budget Limit Met or Exceeded       #")
                    self.logger.info("#            BAREFOOT Calculation Completed             #")
                    self.logger.info("#                                                       #")
                    self.logger.info("#########################################################")
                    start_process = False
                
                self.currentIteration += 1
                self.tmIterCount += 1
    
        
    def __kg_calc_clustering(self, kg_output):
        # convert to a numpy array for ease of indexing
#        self.logger.debug("Clustering Algorithm, input shape: {}".format(kg_output.shape))
        kg_output = np.array(kg_output, dtype=object)
#        self.logger.debug("Clustering Algorithm, kg_output array shape: {}".format(kg_output.shape))
#        df = pd.DataFrame(kg_output)
#        df.to_csv("kg_output.csv")
        # print(kg_output)
        # print(kg_output.shape)
        point_selection = {}
        self.logger.debug("Extract Points for Clustering from Acquisition Function Evaluations")
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
        self.logger.debug("Cluster Acquistion Function Evaluations | {}".format(med_input.shape))
        if med_input.shape[0] > self.batchSize:
            medoids, clusters = k_medoids(med_input[:,0:3], self.batchSize)
        else:
            medoids, clusters = k_medoids(med_input[:,0:3], 1)       
        
        # next, need to get the true values for each of the medoids and update the
        # models before starting next iteration.
        self.logger.debug("Extract True Values for Medoids")
        medoid_index = []
        for i in range(len(medoids)):
            medoid_index.append(int(med_input[medoids[i],3]))
        medoid_out = kg_output[medoid_index,:]
        self.logger.info("Clustering of Acquisition Function Evaluations Completed")
        return medoid_out       
    
    def __start_subprocesses__(self, subprocess_count):
        try:
            os.mkdir('{}/subprocess'.format(self.workingDir))
            self.logger.debug("Subprocess Directory Created")
        except FileExistsError:
            self.logger.debug("Subprocess Directory Already Exists")
            pass
        try:
            os.mkdir('{}/subprocess/LSFOut'.format(self.workingDir))
            self.logger.debug("LSFOut Directory Created")
        except FileExistsError:
            self.logger.debug("LSFOut Directory Already Exists")
            pass
        # This string is used to create the job files for the subprocesses used when calculating the knowledge gradient
        with open("{}/data/processStrings".format(self.workingDir), 'rb') as f:
            processStrings = load(f)
        
        self.logger.info("Strings for Subprocess Shell Files Loaded")
        
        subProcessStr = processStrings[0]
        runProcessStr = processStrings[1]
        calculation_count = self.sampleCount*self.hpCount*(len(self.ROM))
        if calculation_count % subprocess_count == 0:
            calcPerProcess = int(calculation_count/subprocess_count)
        else:
            calcPerProcess = int(calculation_count/subprocess_count) + 1
        
        self.logger.info("{} Subprocess Jobs | {} Calculations per Subprocess".format(subprocess_count, calcPerProcess))
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
        self.logger.info("Waiting for Subprocess Jobs to start")
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
                self.logger.info("All Subprocess Jobs Started Successfully")
            # waiting for 2 hours for all the subprocesses to start will stop the waiting
            # and return false from this function to say that all the processes weren't
            # started yet
            if count == 240:
                all_pending = False
                self.logger.critical("Subprocess Jobs Outstanding after 2 Hours | {}/{} Jobs Started".format(total_started, subprocess_count))
                
        return calcPerProcess, all_started

    def __update_reduced_order_models__(self):
        self.logger.info("Recalculate all evaluated points for ROM to ensure correct model results are used")
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
        self.logger.info("Create New Reification Object")
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
        self.logger.info("New Evaluations Saved | Reification Object Updated")
        pass
    
    def __external_TM_data_save(self, TMEvaluationPoints, count):
        outputData = np.zeros((len(TMEvaluationPoints), self.nDim+1))
        for ii in range(len(TMEvaluationPoints)):
            outputData[ii,0:self.nDim] = TMEvaluationPoints[ii]["Input Values"]
            
        colNames = self.inputLabels.append("y")
        outputData = pd.DataFrame(outputData, columns=colNames)
        outputData.to_csv('{}/results/{}/TruthModelEvaluationPoints.csv'.format(self.workingDir, 
                                                                                self.calculationName))
        with open('{}/results/{}/countData'.format(self.workingDir, self.calculationName), 'wb') as f:
            dump(count, f)
        self.__save_calculation_state()
        self.logger.critical("Truth Model Evaluation Points Copied to File | Restart Process when results are ready")
    
    def __external_TM_data_load(self):
        self.__load_from_save()
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
        
        self.logger.info("Truth Model Evaluations Loaded")
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
        
        if np.max(temp_y) > self.maxTM:
            self.maxTM = np.max(temp_y)
            
        self.__add_to_iterationData(time()-self.timeCheck, count)
        self.timeCheck = time()
        
        if self.updateROMafterTM:
            self.__update_reduced_order_models__()

        self.__save_output_dataframes()
        self.__save_calculation_state()
        self.logger.info("Iteration {} Completed Successfully".format(self.currentIteration))
        self.currentIteration += 1
        self.tmIterCount += 1
        
        
        



##############################################################################
##############################################################################
##                                                                          ##
##                        Test Code Section                                 ##
##                                                                          ##
##############################################################################
##############################################################################

import matplotlib.pyplot as plt

def rom1(x):
    x = x*(2)+0.5
    return -np.sin(9.5*np.pi*x) / (2*x)

def rom2(x):
    x = x*(2)+0.5
    return -(x-1)**4

def tm(x):
    x = x*(2)+0.5
    # Gramacy & Lee Test Function
    return -(x-1)**4 - np.sin(10*np.pi*x) / (2*x)

def plot_results(calcName):
    x = np.linspace(0,1,1000)

    y1 = tm(x)
    y2 = rom1(x)
    y3 = rom2(x)
    
    plt.figure()
    plt.plot(x,y1,label="TM")
    plt.plot(x,y2,label="ROM1")
    plt.plot(x,y3,label="ROM2")
    plt.legend()

    with open('./results/{}/iterationData'.format(calcName), 'rb') as f:
        iterationData = load(f)
    
    plt.figure()
    plt.plot(iterationData.loc[:,"Iteration"], iterationData.loc[:,"Max Found"])

def singeNodeTest():
    # np.random.seed(100)
    ROMList = [rom1, rom2]
    test = barefoot(ROMModelList=ROMList, TruthModel=tm, 
                    calcInitData=True, initDataPathorNum=[1,1,1,1], nDim=1, 
                    calculationName="SingleNodeTest", acquisitionFunc="EI")
    modelParam = {'model_l':[[0.1],[0.1]], 
                'model_sf':[1,1,1], 
                'model_sn':[0.01,0.01], 
                'means':[0,0], 
                'std':[1,1], 
                'err_l':[[0.1],[0.1]], 
                'err_sf':[1,1,1], 
                'err_sn':[0.01,0.01],
                'costs':[1,2,20]}
    test.initialize_parameters(modelParam=modelParam, iterLimit=30, 
                                sampleCount=10, hpCount=50, 
                                batchSize=2, tmIter=5)
    test.run_optimization()
    
    plot_results("SingleNodeTest")

if __name__ == "__main__":
    singeNodeTest()