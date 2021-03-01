# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 06:17:35 2021

@author: Richard Couperthwaite
"""

from pickle import dump, load
import concurrent.futures
import numpy as np
from reificationFusion import model_reification
from acquisitionFunc import knowledge_gradient, expected_improvement
from sys import argv
from time import sleep, time
from multiprocessing import cpu_count
import logging

# create logger with 'spam_application'
logger = logging.getLogger('BAREFOOT.subprocess')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('BAREFOOT.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(fh)


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

if __name__ == "__main__":
    param = argv

    logger.info("Subprocess {} | started".format(param[1]))
        
    with open("subprocess/sub{}.start".format(param[1]), 'w') as f:
        f.write("subprocess started successfully\n\n") 
    
    not_close = True

    while not_close:
        try:
            with open("subprocess/sub{}.control".format(param[1]), 'rb') as f:
                control_param = load(f)
            with open("subprocess/sub{}.start".format(param[1]), 'a') as f:
                f.write("Control File Found - {} | {}\n".format(control_param[0], control_param[1]))
                logger.debug("Control File Found - {} | {}\n".format(control_param[0], control_param[1]))
                
            if control_param[0] == 0:
                logger.info("{} | New Subprocess calculation started\n".format(param[1]))

                if control_param[2] == "KG":
                    function = calculate_KG
                elif control_param[2] == "EI":
                    function = calculate_EI
                
                start = time()
                
                if control_param[1] == "iteration":
                    with open("subprocess/{}.dump".format(param[1]), 'rb') as f:
                        parameters = load(f)
                    logger.debug("{} | Reduced Order Model Calculation Started | {} Calculations".format(param[1], len(parameters)))
                    kg_output = []
                    count = 0
                    with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                        for result_from_process in zip(parameters, executor.map(function,parameters)):
                            params, results = result_from_process
                            kg_output.append(results)
                            count += 1
                            logger.debug("{} | {} / {} Calculations Completed".format(param[1], count, len(parameters)))
                            
                    with open("subprocess/{}.output".format(param[1]), 'wb') as f:
                        dump(kg_output, f)
                    
                    
                elif control_param[1] == "fused":
                    with open("subprocess/fused.dump", 'rb') as f:
                        parameters = load(f)
                    logger.debug("{} | Fused Model Calculation Started | {} Calculations".format(param[1], len(parameters)))
                    fused_output = []
                    count = 0
                    with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                        for result_from_process in zip(parameters, executor.map(fused_calculate,parameters)):
                            params, results = result_from_process
                            fused_output.append(results)
                            count += 1
                            logger.debug("{} | {} / {} Calculations Completed".format(param[1], count, len(parameters)))

                    max_values = np.zeros((len(parameters[0][5],2)))
                    
                    for ii in range(len(fused_output)):
                        if max_values[fused_output[ii,1],0] != 0:
                            if max_values[fused_output[ii,1],0] < fused_output[ii,0]:
                                max_values[fused_output[ii,1],0] = fused_output[ii,0]
                                max_values[fused_output[ii,1],1] = fused_output[ii,1]
                        else:
                            max_values[fused_output[ii,1],0] = fused_output[ii,0]
                            max_values[fused_output[ii,1],1] = fused_output[ii,1]
                                
                    fused_output = max_values[np.where(max_values[:,0]>0)]

                    with open("subprocess/fused.output", 'wb') as f:
                        dump(fused_output, f)
                    
            
                with open("subprocess/sub{}.control".format(param[1]), 'wb') as f:
                    control_param[0] = 1
                    dump(control_param, f)
                
                logger.info("{} | Calculation Results Dumped | {} hours\n".format(param[1], np.round((time()-start)/3600, 4)))
            
        except:
            pass
        
        sleep(30)

        try:
            with open('subprocess/close{}'.format(param[1]), 'r') as f:
                d = f.read()
            not_close = False
            logger.debug("{} | Close Command Found".format(param[1]))
        except FileNotFoundError:
            pass
            
    with open("subprocess/sub{}.start".format(param[1]), 'a') as f:
        f.write("subprocess finished successfully\n\n")
    logger.info("{} | Subprocess Finished".format(param[1]))