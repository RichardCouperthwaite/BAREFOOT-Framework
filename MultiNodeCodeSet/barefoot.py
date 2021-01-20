# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:48:26 2020

@author: richardcouperthwaite
"""

import numpy as np
from copy import deepcopy
import scipy
import os
from time import sleep, time
from pyDOE import lhs
from pickle import dump, load
from kmedoids import kMedoids
from reificationFusion import model_reification
import subprocess

                   
def k_medoids(sample, num_clusters):
    D = scipy.spatial.distance_matrix(sample, sample)
    M, C = kMedoids(D, num_clusters)
    return M, C  

def batch_optimization(param, ndim, fused_points, initial_data, models, low_bound, \
                       upper_bound, model_param, load_from_save):
    """
    Batch optimization function for case where the GP parameters for the low 
    order models are known.
    Inputs:
        param: list of command line inputs
                Index:
                    0: not used
                    1: choice of Kernel
                    2: Number of iterations to run the optimizer
                    3: Number of samples to test on
                    4: Number of hyper-parameter sets to use
                    5: Number of clusters to use
                    6: Number of iterations between Truth Model calls
                    7: Total Budget to cut off optimization at
                    8: Budget to expend before calling the Truth Model
                    9: Label for the optimization run
        ndim: Number of dimensions in the input data
        fused_points: number of points to used for each dimension in the fusion process
        initial_data: a set of initial x data points to be used to initialise the models
        models: Python list of model functions (Index zero must be the truth model)
        low_bound: Lower bound to the Fused GP hyperparameters
        upper_bound: Upper bound to the fused GP hyperparameters
        Model_param: A Python Dictionary with the hyperparameters for the low order models
    """
    try:
        os.mkdir('subprocess')
    except FileExistsError:
        pass
    # This string is used to create the job files for the subprocesses used when calculating the knowledge gradient
    with open("data/processStrings", 'rb') as f:
        processStrings = load(f)
    
    subProcessStr = processStrings[0]
    runProcessStr = processStrings[1]
    calculation_count = int(param[3])*int(param[4])*(len(models)-1)
    if calculation_count % 1000 == 0:
        subprocess_count = int(calculation_count/1000)
    else:
        subprocess_count = int(calculation_count/1000) + 1
    # Start all subprocesses
    for fname in range(subprocess_count):
        with open("subprocess/{}.sh".format(fname), 'w') as f:
            f.write(subProcessStr.format(fname))
        with open("subprocess/submit{}.sh".format(fname), 'w') as f:
            f.write(runProcessStr.format(fname))
        
        os.chmod("subprocess/submit{}.sh".format(fname), 0o775)
        subprocess.run(["subprocess/submit{}.sh".format(fname)], shell=True)
    # wait for all subprocesses to start
    all_pending = True
    with open("process_check.txt", "w") as f:
        f.write("***Start Subprocesses***\n")
    while all_pending:
        sleep(300)
        total_started = 0
        for fname in range(subprocess_count):
            if os.path.exists("subprocess/sub{}.start".format(fname)):
                total_started += 1
                with open("process_check.txt", "a") as f:
                    f.write("*Subprocess {} - Running \n".format(fname))
            else:
                with open("process_check.txt", "a") as f:
                    f.write("*Subprocess {} - Pending \n".format(fname))

        if total_started == subprocess_count:
            all_pending = False
    
    # Separate the input values into individual variables.
    kernel = param[1] #'M52'      # define the kernel to be used as the Matern 5/2 kernel
    iter_count = int(param[2]) #201     # define the number of iterations
    sample_count = int(param[3]) #50   # define the number of samples to test on (modified by classifier)
    hp_count = int(param[4]) #500       # define the number of hyper-parameter sets
    num_medoids = int(param[5]) #2     # define the number of medoid clusters to use
    tm_iter = int(param[6]) #10        # define the number of iterations between each RVE call
    total_budget = int(param[7])
    tm_budget = int(param[8])
    date = param[9]
    # Create the results directories if they do not exist
    try:
        os.mkdir('results')
    except FileExistsError:
        pass
    try:
        os.mkdir('results/{}'.format(date))
    except FileExistsError:
        pass
        
    # define the points for creating the fused GP
    points1 = fused_points
    temp = np.linspace(0,1,points1,endpoint=True)
    x_fused = np.zeros((points1**ndim,ndim))
    counter = np.zeros((ndim))
    tm_Model_Max = -1e6
    
    for i in range(points1**ndim):
        new_array = np.zeros((ndim))
        for j in range(ndim):
            new_array[j] = temp[int(counter[j])]
        x_fused[i,:] = new_array
        counter[-1] += 1
        for k in range(ndim):
            if counter[-(k+1)] == points1:
                counter[-(k+1)] = 0
                try:
                    counter[-(k+2)] += 1
                except IndexError:
                    pass
    
    # If this is a continuation of a previous calculation, load the last save state
    
    if load_from_save[0]:
        try:
            with open('data/{}_save_state'.format(date), 'rb') as f:
                state = load(f)
            ii = state['iteration'] 
            if state['total budget'] <= 0:
                total_Budget_Left = total_budget
            else:
                total_Budget_Left = state['total budget']
            tm_Budget_Left = state['tm_budget']
            tm_iter_count = state['tm_iter']
            model_control = state['model_control']
            max_TM = state["maxTM"]
            model_record = state["model_record"]
            all_TM_x = ['all_TM_x']
            all_TM_y = ['all_TM_y']
            load_failed = False
            start = time()
        except FileNotFoundError:
            load_failed = True
        
    # If the calculation is to use the same fused model hyperparameters as before
    # then load those, otherwise, generate new ones
    if load_from_save[0] and load_from_save[1]:
        fused_model_HP = state['fused_HP']
    else:
        # Define the Possible hyper parameters
        # due to the fact that distance in the region of 0 to 1 is much "smaller" 
        # than the region above 1, split the number of data points
        # only do this if the upper bound is above 1
        if upper_bound > 1:
            midway = (hp_count - (hp_count % 2))/2
            lower = np.linspace(low_bound, 1.0, num=int(midway), endpoint=False)
            upper = np.linspace(1.0, upper_bound, num=int(midway)+int(hp_count % 2), endpoint=True)
            all_HP = np.append(lower, upper)
        else:
            all_HP = np.linspace(low_bound, upper_bound, num=hp_count, endpoint=True)
        
        # Next, select the hyper-parameter sets by randomly combining the possible
        # hyper-parameters defined above.
        fused_model_HP = np.zeros((hp_count,ndim+1))
        for i in range(hp_count):
            for j in range(ndim+1):
                fused_model_HP[i,j] = all_HP[np.random.randint(0,hp_count)]
    
    # If this is a new calculation, initialize all the values
    if (not load_from_save[0]) or load_failed:
        # Define the Possible hyper parameters
        # due to the fact that distance in the region of 0 to 1 is much "smaller" 
        # than the region above 1, split the number of data points
        # only do this if the upper bound is above 1
        if upper_bound > 1:
            midway = (hp_count - (hp_count % 2))/2
            lower = np.linspace(low_bound, 1.0, num=int(midway), endpoint=False)
            upper = np.linspace(1.0, upper_bound, num=int(midway)+int(hp_count % 2), endpoint=True)
            all_HP = np.append(lower, upper)
        else:
            all_HP = np.linspace(low_bound, upper_bound, num=hp_count, endpoint=True)
        
        # Next, select the hyper-parameter sets by randomly combining the possible
        # hyper-parameters defined above.
        fused_model_HP = np.zeros((hp_count,ndim+1))
        for i in range(hp_count):
            for j in range(ndim+1):
                fused_model_HP[i,j] = all_HP[np.random.randint(0,hp_count)]
        
        # Obtain the initial data for the models and add them to a list of x
        # and y results to be input into the reification class
        x_init = []
        y_init = []
        
        for i in range(len(models)-1):
            x_init.append(initial_data)
            y_init.append(models[i+1](initial_data))
    
    
        truth_output = models[0](initial_data)
        
        tm_Model_Max = np.max(truth_output)
        
        # print(initial_data)
        
        # next we need to create the model reification object with the low order models

        
        model_control = model_reification(x_init, y_init, model_param['model_l'], 
                                          model_param['model_sf'], 
                                          model_param['model_sn'], 
                                          model_param['means'], 
                                          model_param['std'], 
                                          model_param['err_l'], 
                                          model_param['err_sf'], 
                                          model_param['err_sn'], 
                                          initial_data, 
                                          truth_output, len(models)-1, ndim, kernel)
     
        # need to initialise the output files
        # first is the model record that will record how many times each model is 
        # called for each iteration
        model_record = []
        with open("results/{}/model_record.csv".format(date), 'w') as f:
            for iii in range(len(models)):
                f.write("0,")
                model_record.append(0)
            f.write("\n")
            
        all_TM_x = []
        all_TM_y = []
        max_TM = [np.max(truth_output)]
        
        # Create a file to hold the data of what points are being queried from each
        # of the individual models at each iteration. Add the initial data with 
        # index -1 to indicate that this is the starting data
        with open("results/{}/iteration_data.csv".format(date), 'w') as f:
            init_str = "Iteration,Model,"
            for kk in range(ndim):
                init_str += 'x{},'.format(kk)
            f.write(init_str + "Model Output,\n")
            for iii in range(len(models)-1):
                for jjj in range(initial_data.shape[0]):
                    f.write("{},{},".format(-1,iii))
                    if ndim > 1:
                        for kkk in range(ndim):
                            f.write("{},".format(initial_data[jjj,kkk]))
                    else:
                        f.write("{},".format(initial_data[jjj]))
                    f.write("{},\n".format(y_init[iii][jjj]))
            
            for jjj in range(initial_data.shape[0]):
                f.write("{},{},".format(-1,len(models)-1))
                if ndim > 1:
                    for kkk in range(ndim):
                        f.write("{},".format(initial_data[jjj,kkk]))
                else:
                    f.write("{},".format(initial_data[jjj]))
                f.write("{},\n".format(truth_output[jjj]))
        
        # Initialise the file to save the iteration costs.           
        with open("results/{}/iteration_cost.csv".format(date), 'w') as f:
            f.write("Model Cost, Total Budget Left, Truth Model Budget Left,\n")
        # Initialise the log file that will record how many iterations have been completed
        # so that this can be tracked when the number of iterations are not being
        # output in the console
        with open("results/{}/log.txt".format(date), 'w') as f:
            f.write("Iterations Completed,\n")
            
        # initialise the budget values
        tm_Budget_Left = tm_budget
        total_Budget_Left = total_budget
        
        # initialise a counter for the iteration number
        ii = 0
        tm_iter_count = 0        
    
    # The code uses a while loop to account for situations where the budget control
    # is being used. The loop will terminate when either the iteration or
    # budget termination criteria is met
    while True:
        # Create a log file so that the number of iterations completed can be 
        # easily observed while the calculation is running
        with open("results/{}/log.txt".format(date), 'a') as f:
            if ii == 0:
                f.write("{},{}\n".format(ii, 0))
            else:
                f.write("{},{}\n".format(ii, (time()-start)/3600))
        # Get the time of the start of the iteration
        start = time()
        
        # define the test points, it is assumed that the input is on a unit
        # hypercube
        x_test = lhs(ndim, sample_count)
        
        new_mean = []
        
        # obtain predictions from the low-order GPs
        for iii in range(len(models)-1):
            new, var = model_control.predict_low_order(x_test, iii)
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
        for jj in range(len(models)-1):
            for kk in range(sample_count):
                model_temp = deepcopy(model_control)
                model_temp.update_GP(np.expand_dims(x_test[kk], axis=0), 
                                      np.expand_dims(np.array([new_mean[jj][kk]]), 
                                                axis=0), jj)

                for mm in range(hp_count):
                    parameters.append((1, model_temp, x_fused, fused_model_HP[mm,:],
                                    kernel, x_test, jj, kk, mm, sample_count,
                                    model_param['costs'], max_TM[-1]))
                    count += 1
                    if count == 1000:
                        fname = "{}".format(sub_count)
                        sub_fnames.append(fname)
                        
                        with open("subprocess/sub{}.control".format(sub_count), 'wb') as f:
                            control_param = [0, "iteration"]
                            dump(control_param, f)

                        with open("subprocess/{}.dump".format(fname), 'wb') as f:
                            dump(parameters, f)
                        
                        parameters = []
                        count = 0
                        sub_count += 1
        
        if parameters != []:
            fname = "{}".format(sub_count)
            sub_fnames.append(fname)
            
            with open("subprocess/sub{}.control".format(sub_count), 'wb') as f:
                control_param = [0, "iteration"]
                dump(control_param, f)

            with open("subprocess/{}.dump".format(fname), 'wb') as f:
                dump(parameters, f)

        with open("process_check.txt", "a") as f:
            f.write("Calculation sent to subprocess\n")
        # Start a timer and wait for 2 hours, since the subprocess calculations
        # are expected to take a little over 2.5 hours
        sleep(1800)
        
        finished = 0
        # Check to see that all subprocesses have finished
        while finished < len(sub_fnames):
            finished = 0
            for sub_name in sub_fnames:
                with open("subprocess/sub{}.control", 'rb') as f:
                    control_param = load(f)
                    if control_param[0] == 1:
                        finished += 1
            sleep(300)
        # as an approximation of the computational cost of the subprocesses
        # multiply the time of the longest running subprocess by the number of
        # subprocesses
        process_cost = (6000)*len(sub_fnames)
        
        kg_output = []
        # Get the results from the subprocesses and clean up the files
        for sub_name in sub_fnames:
            cont_loop = True
            load_failed = True
            timer = 0
            while cont_loop:
                try:
                    with open("subprocess/{}.output".format(sub_name), 'rb') as f:
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
                os.remove("subprocess/{}.output".format(sub_name))
                os.remove("subprocess/{}.dump".format(sub_name))
        
        # convert to a numpy array for ease of indexing
        kg_output = np.array(kg_output, dtype=object)
        # print(kg_output)
        # print(kg_output.shape)
        point_selection = {}
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
                for mm in range(len(models)-1):
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
                
        model_iter_calls = []
        for mm in range(len(models)):
            model_iter_calls.append(0)
        
        # every 20 iterations update the RVE model with all of the medoid
        # positions, for every other iteration, update the individual low
        # order models
        
        model_cost = time()-start + process_cost
        
        tm_Budget_Left -= model_cost
        total_Budget_Left -= model_cost
        
        tm_iter_count += 1
        max_TM.append(max_TM[ii])
        # Obtain the results from the medoids for the lower order models
        cost = model_param['costs']
        for iii in range(len(medoids)):
            x_index = 5+ndim+1
            x_new = np.array(medoid_out[iii,x_index:], dtype=np.float)
            x_new = np.expand_dims(x_new, 0)
            y_new = models[medoid_out[iii,3]+1](x_new)
            model_control.update_GP(x_new, y_new, medoid_out[iii,3])
            model_iter_calls[medoid_out[iii,3]] += 1
            with open("results/{}/iteration_data.csv".format(date), 'a') as f:
                f.write("{},{},".format(ii,medoid_out[iii,3]))
                for jjj in range(ndim):
                    f.write("{},".format(medoid_out[iii,6+ndim+jjj]))
                try:
                    f.write("{},\n".format(y_new[0]))
                except IndexError:
                    f.write("{},\n".format(y_new))
            total_Budget_Left -= cost[medoid_out[iii,3]]
            tm_Budget_Left -= cost[medoid_out[iii,3]]
        
        for jjj in range(len(models)):
            model_record[jjj] += model_iter_calls[jjj]
            
        with open("results/{}/model_record.csv".format(date), 'a') as f:
            for jjj in range(len(models)):
                f.write("{},".format(model_record[jjj]))
            f.write("\n")
            
        with open("results/{}/iteration_cost.csv".format(date), 'a') as f:
            f.write("{},{},{},\n".format(model_cost, total_Budget_Left, tm_Budget_Left))
        
        
        if (tm_Budget_Left < 0) or (tm_iter_count == tm_iter):
            tm_iter_count = 0
            max_new = 0
            
            # create a test set that is dependent on the number of dimensions            
            tm_test = lhs(ndim, samples=5000*ndim)
            
            parameters = []
            
            # initialize the parameters for the fused model calculations and
            # start the calculation
            for mm in range(hp_count):
                parameters.append((1, model_control, x_fused, fused_model_HP[mm,:],
                                kernel, tm_test, tm_Model_Max, 0.01))
            with open("subprocess/fused.dump", 'wb') as f:
                dump(parameters, f)
                
            with open("subprocess/sub0.control", 'wb') as f:
                control_param = [0, "fused"]
                dump(control_param, f)
            
            # this calculation is expected to be much faster than the previous set 
            # so only wait a short period for the calculation to complete
            sleep(30)
            
            while True:
                if os.path.exists("subprocess/fused.output"):
                    break
                else:
                    sleep(30)
            # obtain results and cleanup the files
            with open("subprocess/fused.output", 'rb') as f:
                fused_output = load(f)
                
            os.remove("subprocess/fused.dump")
            os.remove("subprocess/fused.output")
        
            # cluster the output from the fused model calculations
            fused_output = np.array(fused_output, dtype=object)
            medoids, clusters = k_medoids(fused_output, num_medoids)
            # Calculate the new Truth Model values and add them to the data
            for iii in range(len(medoids)):
                x_index = 5+ndim+1
                x_new = np.array(medoid_out[iii,x_index:], dtype=np.float)
                x_new = np.expand_dims(x_new, 0)
                y_new = models[0](x_new)
                all_TM_x.append(x_new)
                all_TM_y.append(y_new)
                if y_new > max_new:
                    max_new = y_new
                model_control.update_truth(x_new, y_new)
                model_iter_calls[len(models)-1] += 1
                with open("results/{}/iteration_data.csv".format(date), 'a') as f:
                    f.write("{},{},".format(ii,medoid_out[iii,3]))
                    for jjj in range(ndim):
                        f.write("{},".format(medoid_out[iii,6+ndim+jjj]))
                    try:
                        f.write("{},\n".format(y_new[0]))
                    except IndexError:
                        f.write("{},\n".format(y_new))
                total_Budget_Left -= model_param['Truth Cost']
                tm_Budget_Left = tm_budget
            sample_count += 5
            if max_new > max_TM[ii]:
                max_TM.append(max_new)
            else: 
                max_TM.append(max_TM[ii])
            
        
        ii += 1
        # Save the calculation information at the end of each iteration so that
        # calculation can be restarted with minimum loss of information if a
        # problem arises.
        with open('data/{}_save_state'.format(date), 'wb') as f:
            state = {"iteration": ii,
                     "total budget": total_Budget_Left,
                     "tm_budget": tm_Budget_Left,
                     "tm_iter": tm_iter_count,
                     "fused_HP": fused_model_HP,
                     "model_control": model_control,
                     "maxTM":max_TM,
                     "model_record": model_record,
                     'all_TM_x': all_TM_x,
                     'all_TM_y': all_TM_y}
            dump(state, f)
            print("Save State {} Completed".format(ii))
        # finish the calculation if either the iteration limit or the total 
        # budget has been exceeded
        if (total_Budget_Left < 0) or (ii > iter_count):
            break
            
    # Write a file that records that the calculation completed successfully
    with open("results/{}/code_finished.txt".format(date), 'w') as f:
        f.write("** Code Finished **\n")
        
    for fname in range(subprocess_count):
        with open("subprocess/close{}".format(fname)) as f:
            f.write("Close Subprocess {}".format(fname))