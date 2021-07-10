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
from util import cartesian, call_model, apply_constraints
from util import calculate_KG, calculate_EI, fused_calculate, calculate_TS, calculate_Greedy, calculate_PI, calculate_UCB
from util import calculate_GPHedge, evaluateFusedModel, batchAcquisitionFunc, kmedoids_max
from util import fused_EHVI, calculate_EHVI, Pareto_finder, storeObject
from gpModel import gp_model
from sklearn_extra.cluster import KMedoids
import logging
from pyDOE import lhs

# import concurrent.futures
# def Pool():
#     return concurrent.futures.ProcessPoolExecutor(cpu_count())

from ray.util.multiprocessing import Pool

class barefoot():    
    def __init__(self, ROMModelList=[], TruthModel=[], calcInitData=True, 
                 initDataPathorNum=[], multiNode=0, workingDir=".", 
                 calculationName="Calculation", nDim=1, input_resolution=5, restore_calc=False,
                 updateROMafterTM=False, externalTM=False, acquisitionFunc="KG",
                 A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], keepSubRunning=True, 
                 verbose=False, sampleScheme="LHS", tmSampleOpt="Greedy", logname="BAREFOOT",
                 maximize=True, train_func=[], reification=True, batch=True, 
                 multiObjective=False, multiObjectRef=[]):
        
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
                             reflect in the same function. Requires a training function to be supplied in the 
                             train_func input.
        train_func         : Training function used to retrain the reduced order models after the Truth Model
                             evaluations.
        externalTM         : In cases where it is necessary to evaluate the Truth Model separate to the
                             framework (for example, if the Truth Model is an actual experiment), this toggles
                             the output of the predicted points to a separate file for use externally. The
                             framework is shut down after the data is output, see test examples for how to restart
                             the framework after the external Truth Model has been evaluated
        acquisitionFunc    : The acquisition function to use to evaluate the next best points for the reduced
                             order models. Currently the options are "KG" for Knowledge Gradient and "EI" for expected
                             improvement, "PI" Probability of Improvment, "TS" Thompson sampling, "Greedy" Greedy,
                             "UCB" Upper confidence bound, "Hedge" GP-Hedge Portfolio optimization.
        A, b, Aeq, beq     : Equality and inequality constraints according to the following equations:
                             1) A*x <= b
                             2) Aeq*x == b
        ub, lb             : Upper bounds and lower bounds for inputs, all inputs must receive a value
                             (Specify 0 for lb and 1 for ub if there is no bound for that input)
        func               : function constraints, must take the input matrix (x) and output a vector of length
                             equal to the number of samples in the input matrix (x) with boolean values.
        keepSubRunning     : Determines whether the subprocesses are left running while calling the Truth Model
        verbose            : Determines the logging level for tracking the calculations.
        input_resolution   : How many decimal places to use in the inputs.
        sampleScheme       : Sampling scheme for the test points. Options are "Grid", "LHS", "Custom", "CompFunc".
                             Where the Custom uses preselected test points from a file, and the CompFunc is
                             specifically designed for sampling composition spaces.
        tmSampleOpt        : The acquisition function to use when evaluating next-best points for the Truth Model
        logname            : The name of the log file
        maximize           : Toggles if the problem is a maximization or minimization problem. Default is Maximization.
        reification        : Toggles the use of the multi-fidelity Reification approach
        batch              : Toggles the use of the Batch BO approach
        multiObjective     : Toggles multi-objective optimization
        multiObjectRef     : Holds the reference point required by the EHVI acquisition function
                             
        """
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        
        # create logger to output framework progress
        self.logger = logging.getLogger(logname)
        for h in self.logger.handlers:
            self.logger.removeHandler(h)
        self.logger.setLevel(log_level)
        fh = logging.FileHandler('{}.log'.format(logname))
        fh.setLevel(log_level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add the handler to the logger
        self.logger.addHandler(fh)     
        with open(f"{logname}.log", 'w') as f:
            pass
        
        self.logger.info("#########################################################")
        self.logger.info("#                                                       #")
        self.logger.info("#        Start BAREFOOT Framework Initialization        #")
        self.logger.info("#                                                       #")
        self.logger.info("#########################################################")
        self.logger.info("*********************************************************")
        self.logger.info("* Calculation Name: {}   ".format(calculationName))
        self.logger.info("*********************************************************")
        
        # Restore a previous calculation and restart the timer or load new
        # information and initialize
        
        if restore_calc:
            if externalTM:
                self.__external_TM_data_load(workingDir, calculationName)
            else:
                self.__load_from_save(workingDir, calculationName)
            self.restore_calc = restore_calc
            self.pool = Pool()
            self.timeCheck = time()
            self.logger.info("Previous Save State Restored")
        else:
            self.restore_calc = restore_calc
            self.pool = Pool()
            self.timeCheck = time()
            self.multiObjective = multiObjective
            self.MORef = multiObjectRef
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
            self.maximize = maximize
            if self.multiObjective:
                self.tmSampleOpt = "EHVI"
                self.acquisitionFunc = "EHVI"
                self.logger.warning("Default multiobjective acquisition function (EHVI) selected!")
            else:
                if tmSampleOpt in ["Hedge", "Greedy", "EI", "KG", "TS", "PI", "UCB"]:
                    self.tmSampleOpt = tmSampleOpt
                else:
                    self.tmSampleOpt = "Greedy"
                    self.logger.warning("Invalid Truth Model Acquisition Function! Using default (Greedy).")
                if acquisitionFunc in ["Hedge", "Greedy", "EI", "KG", "TS", "PI", "UCB"]:
                    self.acquisitionFunc = acquisitionFunc
                else:
                    self.acquisitionFunc = "KG"
                    self.logger.warning("Invalid ROM Acquisition Function! Using default (KG).")
            self.nDim = nDim
            self.res = input_resolution
            self.A = A
            self.b = b
            self.Aeq = Aeq
            self.beq = beq
            self.ub = ub
            self.lb = lb
            self.constr_func = func
            self.train_func = train_func
            if sampleScheme in ["LHS", "Grid", "Custom", "CompFunc"]:
                self.sampleScheme = sampleScheme
            else:
                self.sampleScheme = "LHS"
                self.logger.warning("Invalid Sample Scheme! Using default (LHS).")
            if self.multinode != 0:
                self.keepSubRunning = keepSubRunning
            else:
                self.keepSubRunning = True
            self.updateROMafterTM = updateROMafterTM
            self.reification = reification
            self.batch = batch
            self.externalTM = externalTM
            self.__create_dir_and_files()
            self.__create_output_dataframes()
            self.__get_initial_data__()
            self.logger.info("Initialization Completed")  
        
    def __catch_error(func):
        """
        If an error occurs during the initialization of the framework this decorator will catch
        that error
        """
        def close_subs(self, *args, **kwargs):
            no_error = False
            try:
                func(self, *args, **kwargs)
                no_error = True
            except Exception as err:
                self.logger.critical("Initialization Code Failed - See Error Below")
                self.logger.exception(err)
            return no_error
        return close_subs
    
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
            with open("BAREFOOT.log", 'w') as f:
                pass
            if os.path.exists('{}/subprocess'.format(self.workingDir)):
                rmtree('{}/subprocess'.format(self.workingDir))
                self.logger.debug("Existing Subprocess Directory Removed")
            os.mkdir('{}/subprocess'.format(self.workingDir))
            os.mkdir('{}/subprocess/LSFOut'.format(self.workingDir))
            self.logger.debug("Subprocess Directory Created")
                
    def __create_output_dataframes(self):
        # The output of the framework is contained in two pandas dataframes
        # the iterationData df shows the iterations, model calls and maximum 
        # value found
        if self.multiObjective:
            labels2 = ["Iteration", "Calculation Time", "Objective 1", "Objective 2", "Truth Model"]
        else:
            labels2 = ["Iteration", "Max Found", "Calculation Time", "Truth Model"]
        for ii in range(len(self.ROM)):
            labels2.append("ROM {}".format(ii))
        # the evaluatedPoints df contains all the points that have been 
        # evaluated from all models
        if self.multiObjective:
            labels1 = ["Model Index", "Iteration", "y1", "y2"]
        else:
            labels1 = ["Model Index", "Iteration", "y"]
        for ii in range(self.nDim):
            labels1.append("x{}".format(ii))
            if self.multiObjective:
                labels2.append("x{}".format(ii))
            self.inputLabels.append("x{}".format(ii))
            
        self.evaluatedPoints = pd.DataFrame(columns=labels1)
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
        # for the GP Hedge approach, the choice of model for each iteration is
        # also saved to a separate file
        hedge_out = {"ROM":[], "TM":[]}
        if self.acquisitionFunc == "Hedge":
            hedge_out["ROM"] = self.gpHedgeTrack
        if self.tmSampleOpt == "Hedge":
            hedge_out["TM"] = self.gpHedgeTrackTM
        if self.acquisitionFunc == "Hedge" or self.tmSampleOpt == "Hedge":
            with open('{}/results/{}/hedgeRecord'.format(self.workingDir, self.calculationName), 'wb') as f:
                dump(hedge_out, f)
        if self.multiObjective:
            with open('{}/results/{}/paretoRecord{}'.format(self.workingDir, self.calculationName, self.currentIteration), 'wb') as f:
                dump(self.pareto, f)
        self.logger.info("Dataframes Pickled and Dumped to Results Directory")
            
    def __save_calculation_state(self):
        # This function saves the entire barefoot object into a pickle file
        # with open('{}/data/{}_save_state'.format(self.workingDir, self.calculationName), 'wb') as f:
        #     dump(self, f)
        # self.logger.info("Calculation State Saved")
        self.logger.info("Calculation State Save Skipped")
        
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
        if self.multiObjective:
            temp = np.zeros((eval_x.shape[0], self.nDim+4))
            temp[:,0] = modelIndex
            temp[:,1] = self.currentIteration
            temp[:,2] = eval_y[:,0]
            temp[:,3] = eval_y[:,1]
            temp[:,4:] = eval_x[:,0:]
            temp = pd.DataFrame(temp, columns=self.evaluatedPoints.columns)
        else:
            temp = np.zeros((eval_x.shape[0], self.nDim+3))
            temp[:,0] = modelIndex
            temp[:,1] = self.currentIteration
            temp[:,2] = eval_y
            temp[:,3:] = eval_x
            temp = pd.DataFrame(temp, columns=self.evaluatedPoints.columns)
        self.evaluatedPoints = pd.concat([self.evaluatedPoints,temp])
        if self.multiObjective:
            self.pareto = Pareto_finder(np.array(self.evaluatedPoints.iloc[:,2:4]),self.goal)
        self.logger.debug("{} New Points Added to Evaluated Points Dataframe".format(eval_x.shape[0]))
        
    def __add_to_iterationData(self, calcTime, iterData):
        # Adds new data points to the Iteration Data Dataframe
        if self.multiObjective:
            temp = np.zeros((1,5+len(self.ROM)+self.nDim))
            temp[0,0] = self.currentIteration
            temp[0,1] = calcTime
            temp[0,2] = self.maxTM[0]
            temp[0,3] = self.maxTM[1]
            temp[0,4] = iterData[-1]
            temp[0,5:5+len(self.ROM)] = iterData[0:len(self.ROM)]
        else:
            temp = np.zeros((1,4+len(self.ROM)))
            temp[0,0] = self.currentIteration
            temp[0,1] = self.maxTM
            temp[0,2] = calcTime
            temp[0,3] = iterData[-1]
            temp[0,4:] = iterData[0:len(self.ROM)]
        temp = pd.DataFrame(temp, columns=self.iterationData.columns)
        self.iterationData = pd.concat([self.iterationData,temp])
        self.logger.debug("Iteration {} Data saved to Dataframe".format(self.currentIteration))
        
    @__catch_error
    def __get_initial_data__(self):
        # Function for obtaining the initial data either by calculation or by 
        # extracting the data from a file.
        params = []
        count = []
        param_index = 0
        if self.multiObjective:
            self.maxTM = [-np.inf,-np.inf]
        else:
            self.maxTM = -np.inf
        if self.acquisitionFunc == "Hedge":
            self.gpHedgeHist = [[np.random.random()],[np.random.random()],
                                [np.random.random()],[np.random.random()],
                                [np.random.random()],[np.random.random()]]
            self.gpHedgeProb = np.sum(self.gpHedgeHist, axis=1)
            self.gpHedgeTrack = []
        if self.tmSampleOpt == "Hedge":
            self.gpHedgeHistTM = [[np.random.random()],[np.random.random()],
                                [np.random.random()],[np.random.random()],
                                [np.random.random()],[np.random.random()]]
            self.gpHedgeProbTM = np.sum(self.gpHedgeHistTM, axis=1)
            self.gpHedgeTrackTM = []
        if self.multiObjective:
            if type(self.maximize) == list:
                self.goal = np.array([-1,-1])
                if self.maximize[0]:
                    self.goal[0] = 1
                if self.maximize[1]:
                    self.goal[1] = 1
            else:
                if self.maximize:
                    self.goal = np.array([1,1])
                else:
                    self.goal = np.array([-1,-1])
        else:
            if self.maximize:
                self.goal = 1
            else:
                self.goal = -1
        # Check if data needs to be calculated or extracted
        if self.calcInitData:
            self.logger.debug("Start Calculation of Initial Data")
            # obtain LHS initial data for each reduced order model
            if self.reification:
                for ii in range(len(self.ROM)):
                    count.append(0)                
                    initInput, check = apply_constraints(self.initDataPathorNum[ii], 
                                                         self.nDim, resolution=self.res,
                                                          A=self.A, b=self.b, Aeq=self.Aeq, beq=self.beq, 
                                                          lb=self.lb, ub=self.ub, func=self.constr_func, 
                                                          sampleScheme=self.sampleScheme,opt_sample_size=True)
                
                    
                
                    if check:
                        self.logger.debug("ROM {} - Initial Data - All constraints applied successfully".format(ii))
                    else:
                        self.logger.critical("ROM {} - Initial Data - Some or All Constraints Could not Be applied! Continuing With {}/{}".format(ii, initInput.shape[0], self.initDataPathorNum[ii]))
                    
                    for jj in range(initInput.shape[0]):
                        params.append({"Model Index":ii,
                                       "Model":self.ROM[ii],
                                       "Input Values":initInput[jj,:],
                                       "ParamIndex":param_index})
                        param_index += 1
                    self.ROMInitInput.append(np.zeros_like(initInput))
                    if self.multiObjective:
                        self.ROMInitOutput.append(np.zeros((initInput.shape[0],2)))
                    else:
                        self.ROMInitOutput.append(np.zeros(initInput.shape[0]))
            # Obtain LHS initial data for Truth Model
            initInput, check = apply_constraints(self.initDataPathorNum[-1], 
                                                 self.nDim, resolution=self.res,
                                                 A=self.A, b=self.b, Aeq=self.Aeq, beq=self.beq, 
                                                 lb=self.lb, ub=self.ub, func=self.constr_func, 
                                                 sampleScheme=self.sampleScheme,opt_sample_size=True)
            count.append(0)
            if check:
                self.logger.debug("TM - Initial Data - All constraints applied successfully")
            else:
                self.logger.critical("TM - Initial Data - Some or All Constraints Could not Be applied! Continuing With {}/{}".format(initInput.shape[0], self.initDataPathorNum[-1]))
            for jj in range(initInput.shape[0]):
                params.append({"Model Index":-1,
                               "Model":self.TM,
                               "Input Values":initInput[jj,:],
                               "ParamIndex":param_index})
                param_index += 1
            self.TMInitInput = np.zeros_like(initInput)
            if self.multiObjective:
                self.TMInitOutput = np.zeros((initInput.shape[0],2))
            else:
                self.TMInitOutput = np.zeros(initInput.shape[0])
            
            # Calculate all the initial data in parallel
            temp_x = np.zeros((len(params), self.nDim))
            if self.multiObjective:
                temp_y = np.zeros((len(params),2))
            else:
                temp_y = np.zeros(len(params))
            temp_index = np.zeros(len(params))
            pass_calculations = []
            self.logger.debug("Parameters Defined. Starting Concurrent.Futures Calculation")
            try:
                self.pool.terminate()
                self.pool = Pool()
            except AttributeError:
                self.pool = Pool()
            with self.pool as executor:
                for result_from_process in zip(params, executor.map(call_model, params)):
                    par, results = result_from_process 
                    try:
                        test = results.shape
                        if par["Model Index"] != -1:                        
                            self.ROMInitInput[par["Model Index"]][count[par["Model Index"]],:] = par["Input Values"]
                            if self.multiObjective:
                                self.ROMInitOutput[par["Model Index"]][count[par["Model Index"]]] = np.tile(self.goal, (results.shape[0]))*results
                            else:
                                self.ROMInitOutput[par["Model Index"]][count[par["Model Index"]]] = self.goal*results
                            temp_x[par["ParamIndex"],:] = par["Input Values"]
                            if self.multiObjective:
                                temp_y[par["ParamIndex"],:] = self.goal*results
                            else:
                                temp_y[par["ParamIndex"]] = self.goal*results
                            temp_index[par["ParamIndex"]] = par["Model Index"]
                        else:
                            self.TMInitInput[count[par["Model Index"]],:] = par["Input Values"]
                            self.TMInitOutput[count[par["Model Index"]]] = self.goal*results
                            if self.multiObjective:
                                if results[0,0] > self.maxTM[0]:
                                    self.maxTM[0] = results[0,0]
                                if results[0,1] > self.maxTM[1]:
                                    self.maxTM[1] = results[0,1]
                            else:
                                if np.max(results) > self.maxTM:
                                    self.maxTM = np.max(results)
                            temp_x[par["ParamIndex"],:] = par["Input Values"]
                            if self.multiObjective:
                                temp_y[par["ParamIndex"],:] = self.goal*results
                            else:
                                temp_y[par["ParamIndex"]] = self.goal*results
                            temp_index[par["ParamIndex"]] = par["Model Index"]
                        count[par["Model Index"]] += 1
                        pass_calculations.append(par["ParamIndex"])
                    except AttributeError:
                        pass
            self.logger.debug("Concurrent.Futures Calculation Completed")
            if self.multiObjective:
                temp_y = temp_y[pass_calculations,:]
            else:
                temp_y = temp_y[pass_calculations]
            temp_x = temp_x[pass_calculations,:]
            temp_index = temp_index[pass_calculations]
        else:
            # extract the initial data from the file
            self.logger.debug("Start Loading Initial Data from Files")
            with open(self.initDataPathorNum, 'rb') as f:
                data = load(f)

            # extract data from dictionary in file and assign to correct variables
            self.TMInitOutput = data["TMInitOutput"]
            self.TMInitInput = data["TMInitInput"]
            if self.reification:
                self.ROMInitOutput = data["ROMInitOutput"]
                self.ROMInitInput = data["ROMInitInput"]
            
            ROMSize = 0
            for mmm in range(len(self.ROMInitInput)):
                ROMSize += self.ROMInitOutput[mmm].shape[0]
            
            temp_x = np.zeros((self.TMInitOutput.shape[0]+ROMSize, 
                               self.nDim))
            if self.multiObjective:
                temp_y = np.zeros((self.TMInitOutput.shape[0]+ROMSize,2))
            else:
                temp_y = np.zeros(self.TMInitOutput.shape[0]+ROMSize)
            temp_index = np.zeros(self.TMInitOutput.shape[0]+ROMSize)
            
            ind = 0
            if self.reification:
                for ii in range(len(self.ROM)):
                    for jj in range(self.ROMInitOutput[ii].shape[0]):
                        temp_x[ind,:] = self.ROMInitInput[ii][jj,:]
                        if self.multiObjective:
                            temp_y[ind,:] = self.goal*self.ROMInitOutput[ii][jj,:]
                        else:
                            temp_y[ind] = self.goal*self.ROMInitOutput[ii][jj]
                        temp_index[ind] = ii
                        ind += 1
                    count.append(self.ROMInitInput[ii].shape[0])
            for jj in range(self.TMInitOutput.shape[0]):
                temp_x[ind,:] = self.TMInitInput[jj,:]
                if self.multiObjective:
                    temp_y[ind,:] = self.goal*self.TMInitOutput[jj,:]
                    if np.max(temp_y[0,0]) > self.maxTM[0]:
                        self.maxTM[0] = np.max(temp_y[0,0])
                    if np.max(temp_y[0,1]) > self.maxTM[1]:
                        self.maxTM[1] = np.max(temp_y[0,1])
                else:
                    temp_y[ind] = self.TMInitOutput[jj]
                    if self.TMInitOutput[jj] > self.maxTM:
                        self.maxTM = self.TMInitOutput[jj]
                temp_index[ind] = -1
                ind += 1
            count.append(self.TMInitInput.shape[0])
            self.logger.debug("Loading Data From File Completed")
        # Add initial data to dataframes
        iterData = np.array(count)
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        self.__add_to_iterationData(time()-self.timeCheck, iterData)
        self.logger.debug("Initial Data Saved to Dataframes")
        self.timeCheck = time()
    
    @__catch_error
    def initialize_parameters(self, modelParam, covFunc="M32", iterLimit=100,  
                              sampleCount=50, hpCount=100, batchSize=5, 
                              tmIter=1e6, totalBudget=1e16, tmBudget=1e16, 
                              upperBound=1, lowBound=0.0001, fusedPoints=500, 
                              fusedHP=[], fusedSamples=10000):
        """
        This function sets the conditions for the barefoot framework calculations.
        All parameters have default values except the model parameters.

        Parameters
        ----------
        modelParam : dictionary
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
        covFunc : String, optional
            The covariance function to used for the Gaussian Process models.
            Options are Squared Exponential ("SE") Matern 3/2 ("M32") and 
            Matern 5/2 ("M52"). The default is "M32".
        iterLimit : Int, optional
            How many iterations to run the framework calculation before
            terminating. The default is 100.
        sampleCount : Int, optional
            The number of samples to use for the acquisition function calculations.
            The default is 50.
        hpCount : Int, optional
            The number of hyperparameter sets to use. The default is 100.
        batchSize : Int, optional
            The batch size for the model evaluations. The default is 5.
        tmIter : Int, optional
            The number of iterations to complete before querying the Truth Model. 
            The default is 1e6.
        totalBudget : Int/Float, optional
            The total time budget to expend before terminating the calculation. 
            The default is 1e16.
        tmBudget : Int/Float, optional
            The budget to expend before querying the Truth Model. The default 
            is 1e16.
        upperBound : Float, optional
            The upper bound for the hyperparameters. The default is 1.
        lowBound : Float, optional
            The lower bound for the hyperparameters. The default is 0.0001.
        fusedPoints : Int, optional
            The number of points to sample from a LHS sampler at which to 
            evaluate the fused mean and variance for building the fused model. 
            The default is 500.
        fusedHP : List, optional
            Holds the hyperparameters for the fused model if the approach does not
            use the Batch approach
        fusedSamples : Int, optional
            The number of samples to take from the design space for evaluating the fused
            model for determining next-best points from the Truth model.

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
        self.fusedHP = fusedHP
        self.fusedSamples = fusedSamples
        # The numpy linspace module will contract the distance below 1 if there
        # are also values above 1. The approach implemented here avoids that
        # situation
        temp_max = self.lowBound*10
        all_HP = np.linspace(self.lowBound, temp_max, num=self.hpCount)
        while temp_max < self.upperBound:
            temp_min = deepcopy(temp_max)
            temp_max = temp_max*10
            if temp_max > self.upperBound:
                temp_max = self.upperBound
            all_HP = np.append(all_HP, np.linspace(temp_min, temp_max, num=self.hpCount))
        # randomly combine the options for the hyperparameters into the hyperparameter sets
        self.fusedModelHP = np.zeros((self.hpCount,self.nDim+1))
        for i in range(self.hpCount):
            for j in range(self.nDim+1):
                self.fusedModelHP[i,j] = all_HP[np.random.randint(0,all_HP.shape[0])]
        # create the evaluation points for determining the fused mean and
        # variance
        sampleSize = fusedPoints
        if self.sampleScheme == "CompFunc":
            sampleOption = "CompFunc"
        else:
            sampleOption = "LHS"
        self.xFused, check = apply_constraints(sampleSize, 
                                          self.nDim, resolution=self.res,
                                          A=self.A, b=self.b, Aeq=self.Aeq, beq=self.beq, 
                                          lb=self.lb, ub=self.ub, func=self.constr_func,
                                          sampleScheme=sampleOption, opt_sample_size=False)
        
       
        
        if check:
            self.logger.debug("Fused Points - All constraints applied successfully {}/{}".format(self.xFused.shape[0], sampleSize))
        else:
            self.logger.critical("Fused Points - Sample Size NOT met due to constraints! Continue with {}/{} Samples".format(self.xFused.shape[0], sampleSize))
            
        if not self.restore_calc:
            self.logger.debug("Create Reification Object")
            if self.multiObjective:
                self.TMInitOutput = [np.array(self.TMInitOutput)[:,0],
                                     np.array(self.TMInitOutput)[:,0]]
                # build the reification object with the combined inputs and initial values
                if self.reification:
                    self.ROMInitOutput = np.array(self.ROMInitOutput)
                    temp = [[],[]]
                                    
                    for pp in range(self.ROMInitOutput.shape[0]):
                        temp[0].append(self.ROMInitOutput[pp,:,0])
                        temp[1].append(self.ROMInitOutput[pp,:,1])
                    
                    self.reificationObj = [model_reification(self.ROMInitInput, temp[0], 
                                                      self.modelParam['model_l'], 
                                                      self.modelParam['model_sf'], 
                                                      self.modelParam['model_sn'], 
                                                      self.modelParam['means'], 
                                                      self.modelParam['std'], 
                                                      self.modelParam['err_l'], 
                                                      self.modelParam['err_sf'], 
                                                      self.modelParam['err_sn'], 
                                                      self.TMInitInput, self.TMInitOutput[0], 
                                                      len(self.ROM), self.nDim, self.covFunc),
                                           model_reification(self.ROMInitInput, temp[1], 
                                                      self.modelParam['model_l'], 
                                                      self.modelParam['model_sf'], 
                                                      self.modelParam['model_sn'], 
                                                      self.modelParam['means'], 
                                                      self.modelParam['std'], 
                                                      self.modelParam['err_l'], 
                                                      self.modelParam['err_sf'], 
                                                      self.modelParam['err_sn'], 
                                                      self.TMInitInput, self.TMInitOutput[1], 
                                                      len(self.ROM), self.nDim, self.covFunc)]
                else:
                    self.modelGP = [gp_model(self.TMInitInput, self.TMInitOutput[0], 
                                            np.ones((self.nDim)), 1, 0.05, 
                                            self.nDim, self.covFunc),
                                    gp_model(self.TMInitInput, self.TMInitOutput[1], 
                                            np.ones((self.nDim)), 1, 0.05, 
                                            self.nDim, self.covFunc)]
            else:
                # build the reification object with the combined inputs and initial values
                if self.reification:
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
                else:
                    self.modelGP = gp_model(self.TMInitInput, self.TMInitOutput, 
                                            np.ones((self.nDim)), 1, 0.05, 
                                            self.nDim, self.covFunc)
        self.allTMInput = []
        self.allTMOutput = []
        self.tmBudgetLeft = self.tmBudget
        self.totalBudgetLeft = self.totalBudget
        self.currentIteration += 1
        self.tmIterCount = 0
        self.logger.info("Reification Object Initialized. Ready for Calculations")
        
    def __restart_subs(self):
        # This function restarts the sub processes if they have been closed
        # while doing the Truth Model evaluations
        for kk in range(self.multinode):
            try:
                os.remove("{}/subprocess/close{}".format(self.workingDir, kk))
                os.remove("{}/subprocess/sub{}.control".format(self.workingDir, kk))
                os.remove("{}/subprocess/sub{}.start".format(self.workingDir, kk))
                self.logger.debug("Close File {} removed".format(kk))
            except FileNotFoundError:
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
        # This function controls the parameter setup and transfer for the
        # evaluation of the acquisition functions to determine the next best
        # points for evaluating the Reduced Order Models when using subprocesses
        self.logger.info("Set Up Parameters for Acquisition Function Evaluation and submit to Subprocesses")
        parameters = []
        parameterFileData = []
        sub_fnames = []
        count = 0
        sub_count = 0
        parameterIndex = 0
        parameterFileIndex = 0
        # Pickle the reification object to be loaded by each of the subprocesses
        # this reduces the amount of memory that needs to be transferred
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
        # set up the parameters to be used in the calculations
        
        for jj in range(len(self.ROM)):
            for kk in range(self.sampleCount):
                if self.multiObjective:
                    means = [np.expand_dims(np.array([new_mean[jj][0][kk]]), axis=0),
                             np.expand_dims(np.array([new_mean[jj][1][kk]]), axis=0)]
                    model_temp = [means, self.goal, self.MORef, self.pareto[0]]
                else:
                    model_temp = [np.expand_dims(x_test[kk], axis=0), 
                                  np.expand_dims(np.array([new_mean[jj][kk]]), axis=0), 
                                  jj]
                if self.batch:
                    for mm in range(self.hpCount):
                        parameterFileData.append((self.currentIteration+1, model_temp, self.xFused, self.fusedModelHP[mm,:],
                                        self.covFunc, x_test, jj, kk, mm, self.sampleCount,
                                        self.modelParam['costs'], self.maxTM))
                        parameters.append([parameterIndex, parameterFileIndex])
                        parameterIndex += 1
                        
                        # store every 1000 set of parameters in a file for use in the
                        # subprocesses
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
                            
                            # Send the trigger for the subprocess to pick up the data for
                            # the calculations
                            with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                                control_param = [0, "iteration", self.acquisitionFunc]
                                dump(control_param, f)
                            # dump the index for the parameter files for the subprocess
                            # to load
                            with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                                dump(parameters, f)
                            
                            parameters = []
                            count = 0
                            sub_count += 1
                else:
                    parameterFileData.append((self.currentIteration+1, model_temp, self.xFused, self.fusedHP,
                                    self.covFunc, x_test, jj, kk, 0, self.sampleCount,
                                    self.modelParam['costs'], self.maxTM))
                    parameters.append([parameterIndex, parameterFileIndex])
                    parameterIndex += 1
                    
                    # store every 1000 set of parameters in a file for use in the
                    # subprocesses
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
                        
                        # Send the trigger for the subprocess to pick up the data for
                        # the calculations
                        with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                            control_param = [0, "iteration", self.acquisitionFunc]
                            dump(control_param, f)
                        # dump the index for the parameter files for the subprocess
                        # to load
                        with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                            dump(parameters, f)
                        
                        parameters = []
                        count = 0
                        sub_count += 1
        
        
        # dump the last of the parameter datasets
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
        # trigger the last subprocess and dump the index parameters
        if parameters != []:
            fname = "{}".format(sub_count)
            sub_fnames.append(fname)
            
            with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                control_param = [0, "iteration", self.acquisitionFunc]
                dump(control_param, f)

            with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                dump(parameters, f)
                
        self.logger.info("Start Waiting for Results to Complete")
        # the calculations will take some time, so start a sleep timer to wait
        # for a minute before starting to check for results
        calc_start = time()
        sleep(10)
        
        finished = 0
        
        process_costs = np.zeros((len(sub_fnames)))
        # check for finished subprocess calculations, and only continue once
        # all subprcesses calculations are completed
        while finished < len(sub_fnames):
            finished = 0
            proc_count = 0
            for sub_name in sub_fnames:
                with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_name), 'rb') as f:
                    control_param = load(f)
                    if control_param[0] == 1:
                        finished += 1
                        if process_costs[proc_count] == 0:
                            # When a subprocess has completed, record how long
                            # the subprocess ran for. This is the cost of the 
                            # subprocess calculation
                            process_costs[proc_count] = time()-calc_start
            if finished < len(sub_fnames):          
                sleep(10)
        
        self.logger.info("Acquisition Function Evaluations Completed")
        # Calculate the total subprocess cost.
        process_cost = np.sum(process_costs)
        
        # extract all the outputs from the subprocesses and collate them
        # into a single array
        kg_output = []
        for sub_name in sub_fnames:
            cont_loop = True
            load_failed = True
            timer = 0
            while cont_loop:
                try:
                    with open("{}/subprocess/{}.output".format(self.workingDir, sub_name), 'rb') as f:
                        try:
                            sub_output = load(f)
                        except EOFError:
                            raise FileNotFoundError
                    load_failed = False
                    cont_loop = False
                except FileNotFoundError:
                    sleep(10)
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
        # As before, this function calculates the acquisition function values
        # for determining the next best points to be queried from the reduced
        # order models. This function runs the concurrent.futures calculations
        # directly.
        parameters = []
        parameterFileData = []
        count = 0
        parameterIndex = 0
        parameterFileIndex = 0
        self.logger.debug("Set Up Parameters for Acquisition Function Evaluation")
        # Save the current reification object to a file for loading
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
        # Define the parameters for each calculation
        
        for jj in range(len(self.ROM)):
            for kk in range(self.sampleCount):
                if self.multiObjective:
                    means = [np.expand_dims(np.array([new_mean[jj][0][kk]]), axis=0),
                             np.expand_dims(np.array([new_mean[jj][1][kk]]), axis=0)]
                    model_temp = [means, self.goal, self.MORef, self.pareto[0]]
                else:
                    model_temp = [np.expand_dims(x_test[kk], axis=0), 
                                  np.expand_dims(np.array([new_mean[jj][kk]]), axis=0), 
                                  jj]
                if self.batch:
                    for mm in range(self.hpCount):
                        parameterFileData.append((self.currentIteration+1, model_temp, self.xFused, self.fusedModelHP[mm,:],
                                        self.covFunc, x_test, jj, kk, mm, self.sampleCount,
                                        self.modelParam['costs'], self.maxTM))
                        parameters.append([parameterIndex, parameterFileIndex])
                        parameterIndex += 1
                        # save each 1000 parameter sets to a file to reduce the amount of memory used
                        if len(parameterFileData) == 1000:
                            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                                dump(parameterFileData, f)
                            parameterFileData = []
                            parameterFileIndex += 1
                            parameterIndex = 0
                        count += 1
                else:
                    parameterFileData.append((self.currentIteration+1, model_temp, self.xFused, self.fusedHP,
                                    self.covFunc, x_test, jj, kk, 0, self.sampleCount,
                                    self.modelParam['costs'], self.maxTM))
                    parameters.append([parameterIndex, parameterFileIndex])
                    parameterIndex += 1
                    # save each 1000 parameter sets to a file to reduce the amount of memory used
                    if len(parameterFileData) == 1000:
                        with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                            dump(parameterFileData, f)
                        parameterFileData = []
                        parameterFileIndex += 1
                        parameterIndex = 0
                    count += 1
        
        # save the last of the parameters sets
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
        # set which acquistion function will be used
        if self.acquisitionFunc == "EI":
            acqFunc = calculate_EI
        elif self.acquisitionFunc == "KG":
            acqFunc = calculate_KG
        elif self.acquisitionFunc == "TS":
            acqFunc = calculate_TS
        elif self.acquisitionFunc == "PI":
            acqFunc = calculate_PI
        elif self.acquisitionFunc == "UCB":
            acqFunc = calculate_UCB
        elif self.acquisitionFunc == "Hedge":
            acqFunc = calculate_GPHedge
        elif self.acquisitionFunc == "Greedy":
            acqFunc = calculate_Greedy
        elif self.acquisitionFunc == "EHVI":
            acqFunc = calculate_EHVI
        kg_output = []
        # Start the concurrent calculations and return the output array
        self.logger.info("Start Acquisition Function Evaluations for {} Parameter Sets".format(len(parameters)))
        try:
            self.pool.terminate()
            self.pool = Pool()
        except AttributeError:
            self.pool = Pool()
        with self.pool as executor:
            for result_from_process in zip(parameters, executor.map(acqFunc,parameters)):
                params, results = result_from_process
                kg_output.append(results)
        self.logger.info("Acquisition Function Evaluations Completed")
        return kg_output, 0    
    
    def __run_multinode_fused(self, tm_test):
        # As with the reduced order model calculations, this function evaluates
        # the selected acquisition function to determine the next best points to 
        # evaluate from the Truth model
        
        # Since this set of calculations uses only the hyperparameter count,
        # a new calculation is needed to determine how many calculations to
        # do on each subprocess
        calc_limit = (-(-self.hpCount//self.multinode)) 
        self.logger.debug("Define Parameters for Max Value Evaluations")
        parameters = []
        parameterFileData = []
        parameterIndex = 0
        parameterFileIndex = 0
        count = 0
        sub_count = 0
        sub_fnames = []
        # Save the reification object to a file
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
            
        if self.multiObjective:
            extra_data = [self.pareto[0], self.goal, self.MORef]
        else:
            extra_data = []
            
        for mm in range(self.hpCount):
            parameterFileData.append((self.currentIteration+1, extra_data, self.xFused, self.fusedModelHP[mm,:],
                            self.covFunc, tm_test, self.maxTM, 0.01, self.tmSampleOpt))
            parameters.append([parameterIndex, parameterFileIndex])
            parameterIndex += 1
            count += 1
            
            # Save every 500 parameter sets to a separate file to reduce memory
            # usage
            if len(parameterFileData) == 500:
                with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                    dump(parameterFileData, f)
                parameterFileData = []
                parameterFileIndex += 1
                parameterIndex = 0
            
            if count == calc_limit:
                fname = "{}".format(sub_count)
                sub_fnames.append(fname)
                # Trigger the subprocesses with a new calculation set
                with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                    control_param = [0, "fused", self.acquisitionFunc]
                    dump(control_param, f)
                # save the parameter indices to a file
                with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                    dump(parameters, f)
                
                parameters = []
                count = 0
                sub_count += 1
        # save the last of the parameter sets to a file
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
                
        if parameters != []:
            fname = "{}".format(sub_count)
            sub_fnames.append(fname)
            # Trigger the final subprocess to start calculations
            with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_count), 'wb') as f:
                control_param = [0, "fused", self.acquisitionFunc]
                dump(control_param, f)
            # dump the parameter indices to a file
            with open("{}/subprocess/{}.dump".format(self.workingDir, fname), 'wb') as f:
                dump(parameters, f)

        self.logger.info("Parameters for Max Value Calculations Sent to Subprocess")
        # wait for calculations to finish
        sleep(10)

        finished = 0
        # check that all calculations have completed before continuing
        while finished < len(sub_fnames):
            finished = 0
            for sub_name in sub_fnames:
                with open("{}/subprocess/sub{}.control".format(self.workingDir, sub_name), 'rb') as f:
                    control_param = load(f)
                    if control_param[0] == 1:
                        finished += 1
            if finished < len(sub_fnames):          
                sleep(10)        
        
        fused_output = []
        # Extract the outputs from the individual subprocess output files and
        # collate into a single array
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
                    sleep(10)
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
        
        # change the format of the output array to be a numpy array
        fused_output = np.array(fused_output, dtype=object)
        if fused_output.shape[0] == 0:
            fused_output = np.array([[0,0]])
        
        self.logger.info("Max Value Calculations Completed")
        return fused_output
        
    def __run_singlenode_fused(self, tm_test):
        # This function achieves the same functionality as the multi-node fused
        # function above, but does it all on the base node, rather than sending the
        # data to subprocesses.
        parameters = []
        parameterFileData = []
        # initialize the parameters for the fused model calculations and
        # start the calculation
        self.logger.debug("Define Parameters for Max Value Evaluations")
        parameterIndex = 0
        parameterFileIndex = 0
        # save the reification object to a separate file
        with open("data/reificationObj", 'wb') as f:
            dump(self.reificationObj, f)
            
        if self.multiObjective:
            extra_data = [self.pareto[0], self.goal, self.MORef]
        else:
            extra_data = []
            
        if self.batch:
            for mm in range(self.hpCount):
                parameterFileData.append((self.currentIteration+1, extra_data, self.xFused, self.fusedModelHP[mm,:],
                                self.covFunc, tm_test, self.maxTM, 0.01, self.tmSampleOpt))
                parameters.append([parameterIndex, parameterFileIndex])
                parameterIndex += 1
                # Save each set of 500 parameters to a separate file
                if len(parameterFileData) == 500:
                    with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                        dump(parameterFileData, f)
                    parameterFileData = []
                    parameterFileIndex += 1
                    parameterIndex = 0
        else:
            parameterFileData.append((self.currentIteration+1, extra_data, self.xFused, self.fusedHP,
                            self.covFunc, tm_test, self.maxTM, 0.01, self.tmSampleOpt))
            parameters.append([parameterIndex, parameterFileIndex])
            parameterIndex += 1
            # Save each set of 500 parameters to a separate file
            if len(parameterFileData) == 500:
                with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                    dump(parameterFileData, f)
                parameterFileData = []
                parameterFileIndex += 1
                parameterIndex = 0
        # Save the last of the parameter sets to a file
        if len(parameterFileData) != 0:
            with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                dump(parameterFileData, f)
        
        # Set up a list of outputs for each of the results from the acquisition
        # functions if using the GP Hedge approach
        if self.tmSampleOpt == "Hedge":
            fused_out = [[],[],[],[],[],[]]
        else:
            # Create just a single list for when using other Acquisition Functions
            fused_output = []
        self.logger.info("Start Max Value Calculations | {} Sets".format(len(parameters)))
        count = 0
        if self.multiObjective:
            func = fused_EHVI
        else:
            func = fused_calculate
        
        # Run the concurrent processes and save the outputs
        try:
            self.pool.terminate()
            self.pool = Pool()
        except AttributeError:
            self.pool = Pool()
        with self.pool as executor:
            for result_from_process in zip(parameters, executor.map(func,parameters)):
                params, results = result_from_process
                if self.tmSampleOpt == "Hedge":
                    fused_out[0].append(results[0][0])
                    fused_out[1].append(results[0][1])
                    fused_out[2].append(results[0][2])
                    fused_out[3].append(results[0][3])
                    fused_out[4].append(results[0][4])
                    fused_out[5].append(results[0][5])
                else:
                    fused_output.append(results[0])
                count += 1
        # When using the GP Hedge approach the list of outputs are returned
        # as-is
        if self.tmSampleOpt == "Hedge":
            return fused_out
        # when using other acquisition functions process the output to attempt
        # the removal of all duplicates and then return the processed output
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
        
        if fused_output.shape[0] == 0:
            fused_output = np.array([[0,0]])
        
        self.logger.info("Max Value Calculations Completed")
        return fused_output

    def __call_ROM(self, medoid_out, x_val):
        # This function serves to evaluate the Reduced Order Models at the 
        # determined points. This is done in parallel to reduce the time taken
        params = []
        count = np.zeros((len(self.ROM)+1)) 
        if self.multiObjective:
            current = np.array(self.iterationData.iloc[:,4:5+len(self.ROM)])[-1,:]
        else:
            current = np.array(self.iterationData.iloc[:,3:])[-1,:]
        count[0:len(self.ROM)] = current[1:]
        count[-1] = current[0]
        param_index = 0
        # Define the parameter sets needed for each calculation
        self.logger.debug("Define Parameters for ROM Function Evaluations")
        for iii in range(medoid_out.shape[0]):
            params.append({"Model Index":medoid_out[iii,3],
                           "Model":self.ROM[medoid_out[iii,3]],
                           "Input Values":x_val[iii,:],
                           "ParamIndex":param_index})
            param_index += 1

        temp_x = np.zeros((len(params), self.nDim))
        if self.multiObjective:
            temp_y = np.zeros((len(params),2))
        else:
            temp_y = np.zeros(len(params))
        temp_index = np.zeros(len(params)) 
        costs = np.zeros(len(params))
        passed_calcs = []
        # Run the concurrent calculations and extract the results
        self.logger.info("Start ROM Function Evaluations | {} Calculations".format(len(params)))
        try:
            self.pool.terminate()
            self.pool = Pool()
        except AttributeError:
            self.pool = Pool()
        with self.pool as executor:
            for result_from_process in zip(params, executor.map(call_model, params)):
                par, results = result_from_process
                costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                # if the truth function fails to evaluate, it should return false
                # and therefore the results are not included in the output
                try:
                    test = results.shape
                    results_evaluate = True
                    passed_calcs.append(par["ParamIndex"])
                except AttributeError:
                    results_evaluate = False
                # if self.multiObjective:
                #     try:
                #         if results == False:
                #             results_evaluate = False
                #     except ValueError:
                #         results_evaluate = True
                #         passed_calcs.append(par["ParamIndex"])
                # else:
                #     if results != False:
                #         results_evaluate = True
                #         passed_calcs.append(par["ParamIndex"])
                #     else:
                #         results_evaluate = False

                if results_evaluate:
                    if len(results.shape) == 1:
                        results = np.expand_dims(results, axis=0)
                    
                    if self.multiObjective:
                        results = self.goal*results
                        temp_y[par["ParamIndex"],:] = results
                    else:
                        results = self.goal*results
                        temp_y[par["ParamIndex"]] = results
                    temp_x[par["ParamIndex"],:] = par["Input Values"]
                    temp_index[par["ParamIndex"]] = par["Model Index"]
                    if self.multiObjective:
                        self.reificationObj[0].update_GP(par["Input Values"], results[0,0], par["Model Index"])
                        self.reificationObj[1].update_GP(par["Input Values"], results[0,1], par["Model Index"])
                    else:
                        self.reificationObj.update_GP(par["Input Values"], results, par["Model Index"])
                    
                    count[par["Model Index"]] += 1
        # Remove any calculations that failed from the output and save the 
        # data
        temp_x = temp_x[passed_calcs]
        temp_y = temp_y[passed_calcs]
        temp_index = temp_index[passed_calcs]
        return temp_x, temp_y, temp_index, costs, count, len(passed_calcs)
    
    def __call_Truth(self, params, count):
        # This function evaluates the truth model at the points defined by the 
        # framework. The parameters for the calculation are defined elsewhere
        # and this framework just runs the evaluations
        temp_x = np.zeros((len(params), self.nDim))
        if self.multiObjective:
            temp_y = np.zeros((len(params),2))
        else:
            temp_y = np.zeros(len(params))
        temp_index = np.zeros(len(params)) 
        costs = np.zeros(len(params))
        passed_calcs = []
        # Run the concurrent calculations and extract the results
        self.logger.info("Start Truth Model Evaluations | {} Sets".format(len(params)))
        try:
            self.pool.terminate()
            self.pool = Pool()
        except AttributeError:
            self.pool = Pool()
        with self.pool as executor:
            for result_from_process in zip(params, executor.map(call_model, params)):
                par, results = result_from_process
                costs[par["ParamIndex"]] += self.modelCosts[par["Model Index"]]
                # if the truth function fails to evaluate, it should return false
                # and therefore the results are not included in the output
                try:
                    test = results.shape
                    results_evaluate = True
                    passed_calcs.append(par["ParamIndex"])
                except AttributeError:
                    results_evaluate = False
                # if self.multiObjective:
                #     try:
                #         if results == False:
                #             results_evaluate = False
                #     except ValueError:
                #         results_evaluate = True
                #         passed_calcs.append(par["ParamIndex"])
                # else:
                #     if results != False:
                #         results_evaluate = True
                #         passed_calcs.append(par["ParamIndex"])
                #     else:
                #         results_evaluate = False
                
                if results_evaluate:
                    if len(results.shape) == 1:
                        results = np.expand_dims(results, axis=0)
                    if self.multiObjective:
                        results = self.goal*results
                        temp_y[par["ParamIndex"],:] = results
                    else:
                        results = self.goal*results
                        temp_y[par["ParamIndex"]] = results
                    temp_x[par["ParamIndex"],:] = par["Input Values"]
                    temp_index[par["ParamIndex"]] = par["Model Index"]
                    count[par["Model Index"]] += 1
                    if self.multiObjective:
                        if self.reification:
                            self.reificationObj[0].update_truth(par["Input Values"], results[0,0])
                            self.reificationObj[1].update_truth(par["Input Values"], results[0,1])
                        else:
                            self.modelGP[0].update(par["Input Values"], results[0,0], 0.05, False)
                            self.modelGP[1].update(par["Input Values"], results[0,1], 0.05, False)
                    else:
                        if self.reification:
                            self.reificationObj.update_truth(par["Input Values"], results)
                        else:
                            self.modelGP.update(par["Input Values"], results, 0.05, False)
        # Remove any calculations that failed from the output and save the 
        # data
        
        if passed_calcs != []:
            temp_x = temp_x[passed_calcs]
            temp_y = temp_y[passed_calcs]
            temp_index = temp_index[passed_calcs]
            self.logger.info("Truth Model Evaluations Completed")
            self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
            self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
            if self.multiObjective:
                if np.max(temp_y[:,0]) > self.maxTM[0]:
                    self.maxTM[0] = np.max(temp_y[:,0])
                if np.max(temp_y[:,1]) > self.maxTM[1]:
                    self.maxTM[1] = np.max(temp_y[:,1])
            else:
                if np.max(temp_y) > self.maxTM:
                    self.maxTM = np.max(temp_y)
        else:
            self.logger.critical("All Truth Model Evaluations Failed to Produce Results! Continue with no new results.")
        # Return the updated model call counts
        return count
    
    def __singleAcqFuncApproach(self, x_test, new_mean, calcPerProcess):
        # this function is set up to be used in conjunction with the GP Hedge
        # approach to call the required acquisition function calls
        if self.multinode > 0:
            kg_output, process_cost = self.__run_multinode_acq_func(x_test, 
                                                                  new_mean, 
                                                                  calcPerProcess)
        else:
            kg_output, process_cost = self.__run_singlenode_acq_func(x_test, 
                                                     new_mean)
        return kg_output, process_cost
    
    def __gpHedgeApproach(self, x_test, new_mean, calcPerProcess):
        # This function is for using the GP Hedge Portfolio optimization appraoch
        
        # Calculate the probabilities for each acquisition function
        prob = self.gpHedgeProb/np.sum(self.gpHedgeProb)
        # determine the index of the function with the highest probability
        index_Max_prob = np.where(prob == np.max(prob))[0][0]
        self.gpHedgeTrack.append(index_Max_prob)
        # run the individual acquisition function evaluations
        output, process_cost = self.__singleAcqFuncApproach(x_test, new_mean, calcPerProcess)
        # the output will be a list of lists, choose the one corresponding to the 
        # maximum probability
        kg_output = output[index_Max_prob]
        
        clusters = []
        # determine the batch of next best points for all acquisition function
        # outputs for use in calculating the gain later
        for ii in range(6):
            cluster_output = np.array(output[ii])
            # Cluster the acquisition function output
            medoid_out = self.__kg_calc_clustering(cluster_output)
            clusters.append(x_test[medoid_out[:,2].astype(int),:])
        # save the clusters
        with open("data/hedgeClusters", 'wb') as f:
            dump(clusters, f)
        # return the output from the selected function
        return kg_output, process_cost
    
    def __update_Hedge_Probabilities(self, models, x_val):
        # at each iteration when using the GP Hedge approach it is necessary to 
        # calculate the gain associated with each acquisition function
        
        # load the data, which is the clusters from each acquistion function output
        with open("data/hedgeClusters", 'rb') as f:
            clusters = load(f)
        
        parameters = []
        parameterFileData = []
        # initialize the parameters for the fused model calculations and
        # start the calculation
        self.logger.debug("Define Parameters for Max Value Evaluations")
        parameterIndex = 0
        parameterFileIndex = 0
        if self.reification:
            with open("data/reificationObj", 'wb') as f:
                dump(self.reificationObj, f)
        else:
            with open("data/reificationObj", 'wb') as f:
                dump(self.modelGP, f)
        
        # for each set of results, define the parameters and evaluate all the
        # fused model GPs
        for ii in range(6):
            clusters[ii] = np.array(clusters[ii])
            for mm in range(self.hpCount):
                if models == "ROM":
                    parameterFileData.append((1, self.reification, self.xFused, self.fusedModelHP[mm,:],
                                self.covFunc, clusters[ii], self.maxTM, 0.01, ii))
                elif models == "TM":
                    parameterFileData.append((1, self.reification, self.xFused, self.fusedModelHP[mm,:],
                                self.covFunc, clusters[ii], self.maxTM, 0.01, ii))
                parameters.append([parameterIndex, parameterFileIndex])
                parameterIndex += 1
                # save each set of 500 parameters in a file
                if len(parameterFileData) == 500:
                    with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                        dump(parameterFileData, f)
                    parameterFileData = []
                    parameterFileIndex += 1
                    parameterIndex = 0
            # save the last set of parameters in a file
            if len(parameterFileData) != 0:
                with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                    dump(parameterFileData, f)
        # run all the calculations concurrently and obtain the outputs
        fused_output = [[],[],[],[],[],[]]
        count = 0
        try:
            self.pool.terminate()
            self.pool = Pool()
        except AttributeError:
            self.pool = Pool()
        with self.pool as executor:
            for result_from_process in zip(parameters, executor.map(evaluateFusedModel,parameters)):
                params, results = result_from_process
                fused_output[results[0]].append(results[1])
                count += 1
        
        # update the gain for each acquisition function for either the ROM or TM
        if models == "ROM":
            for ii in range(6):
                mean_output = np.mean(np.array(fused_output[ii]).transpose(), axis=1)
                self.gpHedgeHist[ii].append(np.max(mean_output))                
                if len(self.gpHedgeHist[ii]) > 2*self.tmIterLim:
                    self.gpHedgeHist[ii] = self.gpHedgeHist[ii][1:]
                    
            self.gpHedgeProb = np.sum(self.gpHedgeHist, axis=1)
        elif models == "TM":
            for ii in range(6):
                mean_output = np.mean(np.array(fused_output[ii]).transpose(), axis=1)
                self.gpHedgeHistTM[ii].append(np.max(mean_output))
                if len(self.gpHedgeHistTM[ii]) > 2*self.tmIterLim:
                    self.gpHedgeHistTM[ii] = self.gpHedgeHistTM[ii][1:]
                    
            self.gpHedgeProbTM = np.sum(self.gpHedgeHistTM, axis=1)
    
    def __singleAcqFused(self, tm_test):
        # For the GP Hedge appraoch for the Truth Model, this functions
        # calls the individual calculations in either single- or multi-node configuration
        if self.multinode > 0:
            fused_output = self.__run_multinode_fused(tm_test)
        else:
            fused_output = self.__run_singlenode_fused(tm_test)
        return fused_output
    
    def __hedgeFused(self, tm_test):
        # This function controls the use of the GP Hedge appraoch in the calculation        
        # of the next best points for the Truth model
        
        # calculate the most recent probabilities and determine which acquisition
        # function has the maximum probability
        prob = self.gpHedgeProbTM/np.sum(self.gpHedgeProbTM)
        index_Max_prob = np.where(prob == np.max(prob))[0][0]
        self.gpHedgeTrackTM.append(prob)
        
        # obtain the outputs from the acquisition functions
        output = self.__singleAcqFused(tm_test)
        
        fused_output = output[index_Max_prob]
        
        max_values = np.zeros((tm_test.shape[0],2))
        # process the selected output to remove duplicates
        for ii in range(len(fused_output)):
            if max_values[fused_output[ii][1],0] != 0:
                if max_values[fused_output[ii][1],0] < fused_output[ii][0]:
                    max_values[fused_output[ii][1],0] = fused_output[ii][0]
                    max_values[fused_output[ii][1],1] = fused_output[ii][1]
            else:
                max_values[fused_output[ii][1],0] = fused_output[ii][0]
                max_values[fused_output[ii][1],1] = fused_output[ii][1]
                    
        fused_output = max_values[np.where(max_values[:,0]!=0)]
        
        if fused_output.shape[0] == 0:
            fused_output = np.array([[0,0]])
        
        self.logger.info("Max Value Calculations Completed")

        clust = []
        # cluster all the outputs, for the calculation of the gain at the
        # end of the iteration
        for ii in range(6):
            cluster_output = np.array(output[ii], dtype=object)
            # Cluster the acquisition function output
            try:
                if cluster_output.shape[0] > self.batchSize:
                    # medoids, clusters = k_medoids(cluster_output, self.batchSize)
                    # kmedoids = KMedoids(n_clusters=self.batchSize, random_state=0).fit(cluster_output[:,0].reshape((-1,1)))
                    # medoids = kmedoids.medoid_indices_
                    # medoids = kmedoids_max(cluster_output[:,0].reshape((-1,1)), self.batchSize)
                    medoids = kmedoids_max(cluster_output, self.batchSize)
                else:
                    medoids = []
                    for iii in range(cluster_output.shape[0]):
                        medoids.append(iii)
            except:
                # medoids, clusters = k_medoids(cluster_output, 1)
                # kmedoids = KMedoids(n_clusters=self.batchSize, random_state=0).fit(cluster_output[:,0].reshape((-1,1)))
                # medoids = kmedoids.medoid_indices_
                # medoids = kmedoids_max(cluster_output[:,0].reshape((-1,1)), self.batchSize)
                medoids = kmedoids_max(cluster_output, 1)
            clust.append(np.array(tm_test[medoids,:], dtype=np.float))
        # save the clusters for use later
        with open("data/hedgeClusters", 'wb') as f:
            dump(clust, f)
        
        return fused_output
    
    def __close_subs_on_error(func):
        """
        If an error occurs during the optimization, a multinode calculation must
        still close all subprocesses to avoid excessive computing hour costs
        """
        def close_subs(self):
            no_error = False
            try:
                func(self)
                no_error = True
            except Exception as err:
                self.logger.critical("Optimization Code Failed - See Error Below")
                self.logger.exception(err)
                
            if self.multinode > 0:
                for fname in range(self.multinode):
                    with open("{}/subprocess/close{}".format(self.workingDir, fname), 'w') as f:
                        f.write("Close Subprocess {}".format(fname))
            return no_error
        return close_subs
        
    @__close_subs_on_error
    def run_BAREFOOT(self):
        """
        This is the main optimization control function which handles all the calculations
        of the BAREFOOT Framework
        """
        if self.batch:
            self.logger.info("Start Full BAREFOOT Framework Calculation")
        else:
            self.logger.info("Start Reification Only Framework Calculation")
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
                
                # Check constraints and obtain latin-hypercube sampled test points
                evalP = []
                for pp in range(len(self.ROM)):
                    evalP.append(np.array(self.evaluatedPoints.loc[self.evaluatedPoints['Model Index']==pp,self.inputLabels]))
                
                x_test, check = apply_constraints(self.sampleCount, 
                                              self.nDim, resolution=self.res,
                                              A=self.A, b=self.b, Aeq=self.Aeq, beq=self.beq, 
                                              lb=self.lb, ub=self.ub, func=self.constr_func,
                                              sampleScheme=self.sampleScheme,opt_sample_size=True,
                                              evaluatedPoints=evalP)
                
                 
                
                # If constraints can't be satisfied, notify the user in the log
                if check:
                    self.logger.debug("ROM - All constraints applied successfully {}/{}".format(x_test.shape[0], self.sampleCount))
                else:
                    self.logger.critical("ROM - Sample Size NOT met due to constraints! Continue with {}/{} Samples".format(x_test.shape[0], self.sampleCount))
                
                if self.multiObjective:
                    if self.reification:
                        new_mean = []
                        # obtain predictions from the low-order GPs
                        for iii in range(len(self.ROM)):
                            new1, var1 = self.reificationObj[0].predict_low_order(x_test, iii)
                            
                            new2, var2 = self.reificationObj[1].predict_low_order(x_test, iii)
                            new_mean.append([new1, new2])
                
                    else:
                        new_mean = []
                        new1, var1 = self.modelGP[0].predict_var(x_test)
                        new2, var2 = self.modelGP[1].predict_var(x_test)
                        new_mean.append([new1, new2])
                else:
                    if self.reification:
                        new_mean = []
                        # obtain predictions from the low-order GPs
                        for iii in range(len(self.ROM)):
                            new, var = self.reificationObj.predict_low_order(x_test, iii)
                            new_mean.append(new)
                    else:
                        new_mean, var = self.modelGP.predict_var(x_test)
                # Calculate the Acquisition Function for each of the test points in each
                # model for each set of hyperparameters
                
                if self.acquisitionFunc == "Hedge":
                    kg_output, process_cost = self.__gpHedgeApproach(x_test, new_mean, calcPerProcess)
                else:
                    kg_output, process_cost = self.__singleAcqFuncApproach(x_test, new_mean, calcPerProcess)
                
                kg_output = np.array(kg_output, dtype=object)
                
                # Cluster the acquisition function output
                medoid_out = self.__kg_calc_clustering(kg_output)
                
                
                model_cost = time()-self.timeCheck + process_cost
                self.timeCheck = time()
                
                # Call the reduced order models
                temp_x, temp_y, temp_index, costs, count, check = self.__call_ROM(medoid_out, x_test[medoid_out[:,2].astype(int),:])
                
                if check != 0:
                    self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
                    
                    if self.acquisitionFunc == "Hedge":
                        self.__update_Hedge_Probabilities("ROM", x_test)
                else:
                    self.logger.critical("All ROM Evalutions Failed to produce a result! Continue with no new data")

                self.totalBudgetLeft -= np.sum(costs) + model_cost
                self.tmBudgetLeft -= np.sum(costs) + model_cost
                self.logger.info("ROM Function Evaluations Completed")
                
                if (self.tmBudgetLeft < 0) or (self.tmIterCount == self.tmIterLim):
                    self.logger.info("Start Truth Model Evaluations")
                    
                    evalP = [np.array(self.evaluatedPoints.loc[self.evaluatedPoints['Model Index']==-1,self.inputLabels])]
                    
                    # create a test set that is dependent on the number of dimensions            
                    tm_test, check = apply_constraints(self.fusedSamples, 
                                              self.nDim, resolution=self.res,
                                              A=self.A, b=self.b, Aeq=self.Aeq, beq=self.beq, 
                                              lb=self.lb, ub=self.ub, func=self.constr_func, 
                                              sampleScheme=self.sampleScheme, opt_sample_size=True, 
                                              evaluatedPoints=evalP)

                    if check:
                        self.logger.debug("Truth Model Query - All constraints applied successfully")
                    else:
                        self.logger.critical("Truth Model Query - Some or All Constraints Could Not Be Applied! Continuing Without Constraints")
                    
                    # Evaluate the acquistion function to determine the next best
                    # points to evaluate
                    if self.tmSampleOpt == "Hedge":
                        fused_output = self.__hedgeFused(tm_test)
                    else:
                        fused_output = self.__singleAcqFused(tm_test)
                        
                    fused_output = np.array(fused_output)
                    
                    if self.batch:
                        if fused_output.shape[0] > self.batchSize:
                            # medoids, clusters = k_medoids(fused_output[:,0].reshape((-1,1)), self.batchSize)
                            # kmedoids = KMedoids(n_clusters=self.batchSize, random_state=0).fit(fused_output[:,0].reshape((-1,1)))
                            # medoids = kmedoids.medoid_indices_
                            # medoids = kmedoids_max(fused_output[:,0].reshape((-1,1)), self.batchSize)
                            medoids = kmedoids_max(fused_output, self.batchSize)
                        else:
                            if self.batchSize != 0:
                                medoids = []
                                for iii in range(fused_output.shape[0]):
                                    medoids.append(iii)
                        #storeObject([np.where(fused_output[:,0] == np.max(fused_output[:,0]))[0][0], medoids], "ReifiFusedMedoid-{}".format(self.currentIteration))
                    else:
                        max_index = np.where(fused_output[:,0] == np.max(fused_output[:,0]))[0][0]
                        medoids = [max_index]
                               
                    # define the parameters for the Truth Model Evaluations
                    params = []
                    param_index = 0
                    self.logger.debug("Define Parameters for Truth Model Evaluations")
                    for iii in range(len(medoids)):
                        params.append({"Model Index":-1,
                                       "Model":self.TM,
                                       "Input Values":np.array(tm_test[int(fused_output[medoids[iii],1]),:], dtype=np.float),
                                       "ParamIndex":param_index})
                        param_index += 1
                        
                    if len(medoids) < self.batchSize:
                        for iii in range(self.batchSize - len(medoids)):
                            params.append({"Model Index":-1,
                                           "Model":self.TM,
                                           "Input Values":np.array(tm_test[np.random.randint(0,tm_test.shape[0]),:], dtype=np.float),
                                           "ParamIndex":param_index})
                            param_index += 1
                    
                    self.tmIterCount = 0
                    self.tmBudgetLeft = self.tmBudget
                    
                    # If and external Truth Model is used, submit the data for
                    # saving to output
                    if self.externalTM:
                        self.__external_TM_data_save(params, count)
                        break
                    else:
                        # If the subprocesses need to be closed, close them
                        if not self.keepSubRunning:
                            for fname in range(self.multinode):
                                with open("{}/subprocess/close{}".format(self.workingDir, fname), 'w') as f:
                                    f.write("Close Subprocess {}".format(fname))
                                    self.logger.warning("Close Subprocess {}".format(fname))
                        # otherwise, query the Truth Model directly
                        count = self.__call_Truth(params, count)
                    
                    if self.tmSampleOpt == "Hedge":
                        self.__update_Hedge_Probabilities("TM", tm_test)
                    # for multinode calculations, check if subprocesses are being kept
                    # running and restart if not
                    if self.keepSubRunning:
                        pass
                    else:
                        if (self.totalBudgetLeft < 0) or (self.currentIteration >= self.iterLimit):
                            pass
                        else:
                            if self.multinode != 0:
                                self.__restart_subs()
                
                # save the required outputs
                self.__add_to_iterationData(time()-self.timeCheck + model_cost, count)
                self.timeCheck = time()
                self.__save_output_dataframes()
                
                # Update the reduced order models if they need to be retrained
                if (self.tmBudgetLeft < 0) or (self.tmIterCount == self.tmIterLim):
                    if self.updateROMafterTM:
                        self.__update_reduced_order_models__()
                
                
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
      
    @__close_subs_on_error
    def run_BATCH(self):
        self.logger.info("Start Batch Only Framework Calculation")
        start_process = True
        while start_process:
            text_num = str(self.currentIteration)
            self.logger.info("#########################################################")
            self.logger.info("#                Start Iteration : {}                 #".format("0"*(4-len(text_num))+text_num))
            self.logger.info("#########################################################")
            self.timeCheck = time()
            
            # Check constraints and obtain latin-hypercube sampled test points
            evalP = []
            for pp in range(len(self.ROM)):
                evalP.append(np.array(self.evaluatedPoints.loc[self.evaluatedPoints['Model Index']==pp,self.inputLabels]))
            
            x_test, check = apply_constraints(self.sampleCount, 
                                          self.nDim, resolution=self.res,
                                          A=self.A, b=self.b, Aeq=self.Aeq, beq=self.beq, 
                                          lb=self.lb, ub=self.ub, func=self.constr_func,
                                          sampleScheme=self.sampleScheme, opt_sample_size=True, 
                                          evaluatedPoints=evalP)
        
            
            # If constraints can't be satisfied, notify the user in the log
            if check:
                self.logger.debug("ROM - All constraints applied successfully {}/{}".format(x_test.shape[0], self.sampleCount))
            else:
                self.logger.critical("ROM - Sample Size NOT met due to constraints! Continue with {}/{} Samples".format(x_test.shape[0], self.sampleCount))
            
            
            parameters = []
            paramFileData = []
            count = np.zeros((len(self.ROM)+1)) 
            if self.multiObjective:
                current = np.array(self.iterationData.iloc[:,4:5+len(self.ROM)])[-1,:]
                extra_data = [self.pareto[0], self.goal, self.MORef]
            else:
                current = np.array(self.iterationData.iloc[:,3:])[-1,:]
                extra_data = []
            count[0:len(self.ROM)] = current[1:]
            count[-1] = current[0]
            parameterIndex = 0
            parameterFileIndex = 0
            with open("data/reificationObj", 'wb') as f:
                dump(self.modelGP, f)
            
            for jj in range(self.hpCount):
                paramFileData.append((self.currentIteration+1, x_test, self.fusedModelHP[jj,:], 
                                          self.maxTM, self.tmSampleOpt, extra_data)) 
                parameters.append([parameterIndex, parameterFileIndex])
                parameterIndex += 1
                # save each 1000 parameter sets to a file to reduce the amount of memory used
                if len(paramFileData) == 1000:
                    with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                        dump(paramFileData, f)
                    paramFileData = []
                    parameterFileIndex += 1
                    parameterIndex = 0
            
            # dump the last of the parameter datasets
            if len(paramFileData) != 0:
                with open("data/parameterSets/parameterSet{}".format(parameterFileIndex), 'wb') as f:
                    dump(paramFileData, f)
            
            # Set up a list of outputs for each of the results from the acquisition
            # functions if using the GP Hedge approach
            if self.tmSampleOpt == "Hedge":
                prob = self.gpHedgeProbTM/np.sum(self.gpHedgeProbTM)
                index_Max_prob = np.where(prob == np.max(prob))[0][0]
                self.gpHedgeTrackTM.append(prob)
                kg_out = [[],[],[],[],[],[]]
            else:
                # Create just a single list for when using other Acquisition Functions
                kg_output = []
            # Start the concurrent calculations and return the output array
            self.logger.info("Start Acquisition Function Evaluations for {} Parameter Sets".format(len(parameters)))
            try:
                self.pool.terminate()
                self.pool = Pool()
            except AttributeError:
                self.pool = Pool()
            with self.pool as executor:
                for result_from_process in zip(parameters, executor.map(batchAcquisitionFunc,parameters)):
                    params, results = result_from_process
                    
                    if self.tmSampleOpt == "Hedge":
                        kg_out[0].append(results[0])
                        kg_out[1].append(results[1])
                        kg_out[2].append(results[2])
                        kg_out[3].append(results[3])
                        kg_out[4].append(results[4])
                        kg_out[5].append(results[5])
                    else:
                        kg_output.append(results)
            
            self.logger.info("Acquisition Function Evaluations Completed")

            def get_medoids(kg_output):
                # Cluster the acquisition function output
                if kg_output.shape[0] > self.batchSize:
                    medoids = kmedoids_max(kg_output[:,0:3], self.batchSize)
                else:
                    medoids = []
                    for iii in range(kg_output.shape[0]):
                        medoids.append(iii)
                return medoids
            
            if self.tmSampleOpt == "Hedge":
                clust = []
                for pp in range(6):
                    kg_out[pp] = np.array(kg_out[pp])
                    #kg_out[pp][np.isinf(kg_out[pp])] = -1e16
                    
                    kg_out[pp] = np.unique(kg_out[pp], axis=0)

                    med = get_medoids(kg_out[pp])
                    
                    if pp == index_Max_prob:
                        medoids = med
                        kg_output = kg_out[pp]
                    index = np.array(kg_out[pp][med,1],dtype=np.uint8)
                    clust.append(np.array(x_test[index,:], dtype=np.float))
                with open("data/hedgeClusters", 'wb') as f:
                    dump(clust, f)
            else:
                kg_output = np.array(kg_output)
                kg_output = np.unique(kg_output, axis=0)
                medoids = get_medoids(kg_output)

            model_cost = time()-self.timeCheck
            self.timeCheck = time()

            # define the parameters for the Truth Model Evaluations
            params = []
            param_index = 0
            self.logger.debug("Define Parameters for Model Evaluations")
            for iii in range(len(medoids)):
                params.append({"Model Index":-1,
                               "Model":self.TM,
                               "Input Values":np.array(x_test[int(kg_output[medoids[iii],1]),:], dtype=np.float),
                               "ParamIndex":param_index})
                param_index += 1

            count = self.__call_Truth(params, count)
            
            if self.acquisitionFunc == "Hedge":
                self.__update_Hedge_Probabilities("TM", x_test)
            
            # save the required outputs
            self.__add_to_iterationData(time()-self.timeCheck + model_cost, count)
            self.timeCheck = time()
            self.__save_output_dataframes()
            
            # Update the reduced order models if they need to be retrained
            if (self.tmBudgetLeft < 0) or (self.tmIterCount == self.tmIterLim):
                if self.updateROMafterTM:
                    self.__update_reduced_order_models__()
            
            
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
        
    def run_optimization(self):
        
        if self.reification:
            return self.run_BAREFOOT()
        else:
            return self.run_BATCH()

    
    def __kg_calc_clustering(self, kg_output):
        # This function clusters the output from the Reduced Order Model stage 
        # acquistion function evaluations There is some processing required to 
        # obtain the correct format.
        # convert to a numpy array for ease of indexing
        # kg_output = np.array(kg_output, dtype=object)
        point_selection = {}
        self.logger.debug("Extract Points for Clustering from Acquisition Function Evaluations")
        # process the output to obtain the correct format for the clustering
        # (model index, acquisition function value, input index)
        
        for iii in range(kg_output.shape[0]):
            try:
                if int(kg_output[iii,3]) in point_selection[kg_output[iii,2]]['models']:
                    if kg_output[iii,1] > point_selection[kg_output[iii,2]]['nu'][int(kg_output[iii,3])]:
                        point_selection[kg_output[iii,2]]['nu'][int(kg_output[iii,3])] = kg_output[iii,1]
                        point_selection[kg_output[iii,2]]['kg_out'][int(kg_output[iii,3])] = iii
                else:
                    point_selection[kg_output[iii,2]]['models'].append(int(kg_output[iii,3]))
                    point_selection[kg_output[iii,2]]['nu'][int(kg_output[iii,3])] = kg_output[iii,1]
                    point_selection[kg_output[iii,2]]['kg_out'][int(kg_output[iii,3])] = iii
            except KeyError:
                point_selection[kg_output[iii,2]] = {'models':[int(kg_output[iii,3])],
                                                     'nu':[],
                                                     'kg_out':[]}
                for mm in range(len(self.ROM)):
                    point_selection[kg_output[iii,2]]['nu'].append(1e-6)
                    point_selection[kg_output[iii,2]]['kg_out'].append(-1)
                point_selection[kg_output[iii,2]]['nu'][int(kg_output[iii,3])] = kg_output[iii,1]
                point_selection[kg_output[iii,2]]['kg_out'][int(kg_output[iii,3])] = iii
        
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
        if self.batch:
            if med_input.shape[0] > self.batchSize:
                # medoids, clusters = k_medoids(med_input[:,0:3], self.batchSize)
                # kmedoids = KMedoids(n_clusters=self.batchSize, random_state=0).fit(med_input[:,0].reshape((-1,1)))
                # medoids = kmedoids.medoid_indices_
                # medoids = kmedoids_max(med_input[:,0].reshape((-1,1)), self.batchSize)
                medoids = kmedoids_max(med_input[:,0:3], self.batchSize)
            else:
                # medoids, clusters = k_medoids(med_input[:,0:3], 1)  
                # kmedoids = KMedoids(n_clusters=self.batchSize, random_state=0).fit(med_input[:,0].reshape((-1,1)))
                # medoids = kmedoids.medoid_indices_
                # medoids = kmedoids_max(med_input[:,0].reshape((-1,1)), 1)
                medoids = kmedoids_max(med_input[:,0:3], med_input.shape[0])
            #storeObject([np.where(med_input[:,0] == np.max(med_input[:,0]))[0][0], medoids], "ReifiROM-{}".format(self.currentIteration))
        else:
            max_index = np.where(med_input[:,0] == np.max(med_input[:,0]))[0][0]
            medoids = [max_index]
            
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
        # The subprocesses require a separate directory in the main BAREFOOT
        # directory, so these need to be created if they don't exist
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
        # These strings are used to create the job files for the subprocesses used 
        # when running the calculations in multi-node configuration
        with open("{}/data/processStrings".format(self.workingDir), 'rb') as f:
            processStrings = load(f)
        
        self.logger.info("Strings for Subprocess Shell Files Loaded")
        
        # extract the two process strings and calculate how many calculations
        # will be done per subprocess
        subProcessStr = processStrings[0]
        runProcessStr = processStrings[1]
        if self.batch and self.reification:
            calculation_count = self.sampleCount*self.hpCount*(len(self.ROM))
        elif self.batch and not self.reification:
            calculation_count = self.hpCount
        else:
            calculation_count = self.sampleCount*(len(self.ROM))
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
            sleep(10)
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
            # started yet. This is to save on computational hours if there is a problem
            # but this functionality can be disabled if desired.
            if count == 720:
                all_pending = False
                self.logger.critical("Subprocess Jobs Outstanding after 2 Hours | {}/{} Jobs Started".format(total_started, subprocess_count))
                
        return calcPerProcess, all_started

    def __update_reduced_order_models__(self):
        # If the reduced order models are configured to be retrained after more
        # truth model evaluations have been conducted, this function re-evaluates
        # all the evaluated points and reconstructs the reification object with
        # the new values.
        
        self.train_func("results/{}/".format(self.calculationName))
        
        self.logger.info("Recalculate all evaluated points for ROM to ensure correct model results are used")
        self.ROMInitInput = []
        self.ROMInitOutput = []
        TMDataX = self.reificationObj.x_true
        TMDataY = self.reificationObj.y_true
        params = []
        params_truth = []
        count = []
        param_index = 0
        modelIter_record = []
        for jj in range(len(self.ROM)+1):
            count.append(0)
        for jj in range(self.evaluatedPoints.shape[0]):
            modelIter_record.append([self.evaluatedPoints.loc[jj,"Model Index"], self.evaluatedPoints.loc[jj,"Iteration"]])
            if self.evaluatedPoints.loc[jj,"Model Index"] != -1:
                params.append({"Model Index":self.evaluatedPoints.loc[jj,"Model Index"],
                               "Model":self.ROM[self.evaluatedPoints.loc[jj,"Model Index"]],
                               "Input Values":self.evaluatedPoints.loc[jj,self.inputLabels],
                               "ParamIndex":param_index})
            else:
                count[-1] += 1
                params_truth.append({"Model Index":-1,
                               "Model":self.TM,
                               "Input Values":self.evaluatedPoints.loc[jj,self.inputLabels],
                               "ParamIndex":param_index,
                               "Evaluation": self.evaluatedPoints.loc[jj,"y"]})
            param_index += 1
        for ii in range(len(self.ROM)):
            self.ROMInitInput.append(np.zeros_like(self.reificationObj.x_train[ii]))
            self.ROMInitOutput.append(np.zeros_like(self.reificationObj.y_train[ii]))
            
        temp_x = np.zeros((len(modelIter_record), self.nDim))
        temp_y = np.zeros(len(modelIter_record))
        temp_index = np.zeros(len(modelIter_record))
        temp_iter = np.array(modelIter_record)
        
        # Run the evaluations concurrently and store the outputs         
        try:
            self.pool.terminate()
            self.pool = Pool()
        except AttributeError:
            self.pool = Pool()  
        with self.pool as executor:
            for result_from_process in zip(params, executor.map(call_model, params)):
                par, results = result_from_process
                if par["Model Index"] != -1:
                    self.ROMInitInput[par["Model Index"]][count[par["Model Index"]],:] = par["Input Values"]
                    self.ROMInitOutput[par["Model Index"]][count[par["Model Index"]]] = results
                    temp_x[par["ParamIndex"],:] = par["Input Values"]
                    temp_y[par["ParamIndex"]] = results
                    temp_index[par["ParamIndex"]] = par["Model Index"]
                    count[par["Model Index"]] += 1
        for pp in range(len(params_truth)):
            temp_x[params_truth[pp]["ParamIndex"]] = params_truth[pp]["Input Values"]
            temp_y[params_truth[pp]["ParamIndex"]] = params_truth[pp]["Evaluation"]
            temp_index[params_truth[pp]["ParamIndex"]] = -1
        self.logger.info("Create New Reification Object")
        # Recreate the reification object for further calculations
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
        # save the new data
        # Adds new data points to the evaluated datapoints dataframe
        temp = np.zeros((temp_x.shape[0], self.nDim+3))
        temp[:,0] = temp_index
        temp[:,1] = temp_iter
        temp[:,2] = temp_y
        temp[:,3:] = temp_x
        temp = pd.DataFrame(temp, columns=self.evaluatedPoints.columns)
        self.evaluatedPoints = temp
        self.__add_to_iterationData(time()-self.timeCheck, np.array(count))
        self.timeCheck = time()
        self.logger.info("New Evaluations Saved | Reification Object Updated")
    
    
    
    def __external_TM_data_save(self, TMEvaluationPoints, count):
        # When using an external Truth Model, it is necessary to save the next
        # best points for use in the external calculations or experiments
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
    
    def __external_TM_data_load(self, workingDir, calculationName):
        # When restarting the framework after using an external Truth Model
        # the data from the model must be loaded into the framework
        self.__load_from_save(workingDir, calculationName)
        with open('{}/results/{}/countData'.format(self.workingDir, self.calculationName), 'rb') as f:
            count = load(f)
        TMData = pd.read_csv('{}/results/{}/TruthModelEvaluationPoints.csv'.format(self.workingDir, 
                                                                                             self.calculationName))
        TMData = np.array(TMData)
        
        temp_x = np.zeros((TMData.shape[0], self.nDim))
        if self.multiObjective:
            temp_y = np.zeros((TMData.shape[0], 2))
        else:
            temp_y = np.zeros((TMData.shape[0]))
        temp_index = np.zeros((TMData.shape[0]))
        
        for ii in range(TMData.shape[0]):
            temp_x[ii,:] = TMData[ii,0:self.nDim]
            if self.multiObjective:
                temp_y[ii,:] = TMData[ii,self.nDim+1:self.nDim+3]
            else:
                temp_y[ii] = TMData[ii,self.nDim+1]
            temp_index[ii] = -1
            count[-1] += 1
        # After loading the data, the reification object is updated and the new
        # data saved to the normal framework outputs
        self.logger.info("Truth Model Evaluations Loaded")
        self.__add_to_evaluatedPoints(temp_index, temp_x, temp_y)
        temp_y = self.goal*temp_y
        self.reificationObj.update_truth(temp_x, temp_y)
        self.totalBudgetLeft -= self.batchSize*self.modelCosts[-1]
        
        if self.multiObjective:
            if np.max(temp_y[:,0]) > self.maxTM[:,0]:
                self.maxTM[:,0] = np.max(temp_y[:,0])
        else:
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
        self.logger.info("Finished Loading External TM Data")
        
        










##############################################################################
##############################################################################
##############################################################################

def model1c(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return np.array([-80*np.sin(2*x[:,0])/(441*np.pi) - 160*np.sin(4*x[:,0])/(81*np.pi) + np.pi*np.sin(5*x[:,0])/2,
            -48*np.sin(x[:,1])/(1225*np.pi) - 16*np.sin(3*x[:,1])/(81*np.pi) - 240*np.sin(5*x[:,1])/(121*np.pi)]).transpose()
    # return -80*np.sin(2*x[:,0])/(441*np.pi) - 160*np.sin(4*x[:,0])/(81*np.pi) + np.pi*np.sin(5*x[:,0])/2 - 48*np.sin(x[:,1])/(1225*np.pi) - 16*np.sin(3*x[:,1])/(81*np.pi) - 240*np.sin(5*x[:,1])/(121*np.pi)

def model2c(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return np.array([-60*np.sin(2.5*x[:,0])/(441*np.pi) - 160*np.sin(4.2*x[:,0])/(81*np.pi) + np.pi*np.sin(5.3*x[:,0])/2,
            -60*np.sin(0.5*x[:,1])/(1200*np.pi) - 16*np.sin(3.2*x[:,1])/(81*np.pi) - 240*np.sin(5.3*x[:,1])/(121*np.pi)]).transpose()
    # return -60*np.sin(2.5*x[:,0])/(441*np.pi) - 60*np.sin(0.5*x[:,1])/(1200*np.pi) - 160*np.sin(4.2*x[:,0])/(81*np.pi) + np.pi*np.sin(5.3*x[:,0])/2 - 16*np.sin(3.2*x[:,1])/(81*np.pi) - 240*np.sin(5.3*x[:,1])/(121*np.pi)

def model3c(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return np.array([-100*np.sin(1.5*x[:,0])/(400*np.pi) - 160*np.sin(4.5*x[:,0])/(81*np.pi) + np.pi*np.sin(5.5*x[:,0])/2, 
            -36*np.sin(1.5*x[:,1])/(1200*np.pi) - 16*np.sin(3.5*x[:,1])/(81*np.pi) - 240*np.sin(5.5*x[:,1])/(121*np.pi)]).transpose()
    # return -100*np.sin(1.5*x[:,0])/(400*np.pi) - 36*np.sin(1.5*x[:,1])/(1200*np.pi) - 160*np.sin(4.5*x[:,0])/(81*np.pi) + np.pi*np.sin(5.5*x[:,0])/2 - 16*np.sin(3.5*x[:,1])/(81*np.pi) - 240*np.sin(5.5*x[:,1])/(121*np.pi)

def truthmodel(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    x = np.pi*(x*2 - 1)
    return np.array([np.abs(x[:,0])*np.sin(5*x[:,0]), 
            np.abs(x[:,1])*np.sin(6*x[:,1])]).transpose()
    # return np.abs(x[:,0])*np.sin(5*x[:,0]) + np.abs(x[:,1])*np.sin(6*x[:,1])

def runMultiObjective(gg):
    ROMList = [model1c, model2c, model3c]
    framework = barefoot(ROMModelList=ROMList, TruthModel=truthmodel,  
                         calcInitData=False, initDataPathorNum="data/testMOInitData", nDim=2, 
                    #calcInitData=True, initDataPathorNum=[2,2,2,5], nDim=2, 
                    calculationName="bfIMMI-{}".format(gg), acquisitionFunc="EI",
                    restore_calc=False, logname="barefoot", tmSampleOpt="EI", multiNode=0,
                    multiObjective=True, multiObjectRef=np.array([-np.pi, -np.pi]), reification=False)
    
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
    
    framework.initialize_parameters(modelParam=modelParam, covFunc="M32", iterLimit=50,  
                                      sampleCount=25, hpCount=50, batchSize=2, 
                                      tmIter=1, upperBound=1, lowBound=0.0001, fusedPoints=10)
    
    framework.run_optimization()

if __name__ == "__main__":
    runMultiObjective(0)