# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 06:17:35 2021

@author: Richard Couperthwaite
"""

from pickle import dump, load
import concurrent.futures
import numpy as np
from sys import argv
from time import sleep, time
from multiprocessing import cpu_count
from util import calculate_KG, calculate_EI, fused_calculate, calculate_TS, calculate_GPHedge, calculate_Greedy, calculate_PI, calculate_UCB, calculate_EHVI, fused_EHVI
import logging

if __name__ == "__main__":
    """
    This module is used within the BAREFOOT framework to run as a multi-node instance
    This module runs on each of the subprocess nodes and controls the calculations on that
    node.
    """
    param = argv
    
    # log_level = logging.DEBUG
    log_level = logging.INFO
    
    # create logging instance to record progress of the calculations
    logger = logging.getLogger('BAREFOOT.subprocess')   
    logger.setLevel(log_level)
    fh = logging.FileHandler('BAREFOOT.log')
    fh.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(fh)

    logger.info("Subprocess {} | started".format(param[1]))
    
    # Create a file to show that the subprocess has succesfully started
    with open("subprocess/sub{}.start".format(param[1]), 'w') as f:
        f.write("subprocess started successfully\n\n") 
    
    not_close = True
    # keep the code running until it is shut down by the main node
    while not_close:
        try:
            try:
                # the main code will create these files when it is time for a 
                # calculation to be done on the subprocess node
                with open("subprocess/sub{}.control".format(param[1]), 'rb') as f:
                    control_param = load(f)
                with open("subprocess/sub{}.start".format(param[1]), 'a') as f:
                    f.write("Control File Found - {} | {}\n".format(control_param[0], control_param[1]))
                    logger.debug("Control File Found - {} | {}\n".format(control_param[0], control_param[1]))
            except FileExistsError:
                logger.debug("Control File could not be found")
                control_param = [1,1]
            
            # The main node changes the control_param value to 0 to indicate that
            # there is a calculation to complete
            if control_param[0] == 0:
                logger.info("{} | New Subprocess calculation started\n".format(param[1]))
                # The main code also specifies which acquisition function to use
                if control_param[2] == "KG":
                    function = calculate_KG
                elif control_param[2] == "EI":
                    function = calculate_EI
                elif control_param[2] == "TS":
                    function = calculate_TS
                elif control_param[2] == "Hedge":
                    function = calculate_GPHedge
                elif control_param[2] == "Greedy":
                    function = calculate_Greedy
                elif control_param[2] == "PI":
                    function = calculate_PI
                elif control_param[2] == "UCB":
                    function = calculate_UCB
                elif control_param[2] == "EHVI":
                    function = calculate_EHVI


                start = time()
                # there is a difference between the calculations required for the
                # reduced order modesl (iteration) and the truth model (fused)
                if control_param[1] == "iteration":
                    # Parameters for the calculations are determined in the
                    # main node and are saved in .dump files for each subprocess
                    with open("subprocess/{}.dump".format(param[1]), 'rb') as f:
                        parameters = load(f)
                    logger.debug("{} | Reduced Order Model Calculation Started | {} Calculations".format(param[1], len(parameters)))
                    kg_output = []
                    count = 0
                    # Calculations are conducted in parallel using the concurrent.futures appraoch
                    with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                        for result_from_process in zip(parameters, executor.map(function,parameters)):
                            params, results = result_from_process
                            kg_output.append(results)
                            count += 1
                            if count % 200 == 0:
                                logger.info("{} | {} / {} Calculations Completed".format(param[1], count, len(parameters)))
                    # Once calculations are completed, they are saved to the .output file for the main node to retrieve     
                    with open("subprocess/{}.output".format(param[1]), 'wb') as f:
                        dump(kg_output, f)
                    
                    
                elif control_param[1] == "fused":
                    # Parameters for the calculations are determined in the
                    # main node and are saved in .dump files for each subprocess
                    with open("subprocess/{}.dump".format(param[1]), 'rb') as f:
                        parameters = load(f)
                    logger.debug("{} | Fused Model Calculation Started | {} Calculations".format(param[1], len(parameters)))
                    if control_param[2] == "Hedge":
                        fused_out = [[],[],[],[]]
                    else:
                        fused_output = []
                    count = 0
                    
                    if control_param[2] == "EHVI":
                        func = fused_EHVI
                    else:
                        func = fused_calculate
                    
                    # Calculations are conducted in parallel using the concurrent.futures appraoch
                    with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                        for result_from_process in zip(parameters, executor.map(func,parameters)):
                            params, results = result_from_process
                            if control_param[2] == "Hedge":
                                fused_out[0].append(results[0][0])
                                fused_out[1].append(results[0][1])
                                fused_out[2].append(results[0][2])
                                fused_out[3].append(results[0][3])
                            else:
                                fused_output.append(results[0])
                            count += 1
                            if count % 200 == 0:
                                logger.info("{} | {} / {} Calculations Completed".format(param[1], count, len(parameters)))
                    # if the acquisition function approach is the GP Hedge portfolio
                    # optimization approach then the output from this function needs
                    # no further processing. If any of the others are being used,
                    # there is some processing to attempt to remove duplicates
                    if control_param[2] == "Hedge":
                        with open("subprocess/{}.output".format(param[1]), 'wb') as f:
                            dump(fused_out, f)
                    else:
                    
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
                        # Once calculations are completed, they are saved to the .output file for the main node to retrieve
                        with open("subprocess/{}.output".format(param[1]), 'wb') as f:
                            dump(fused_output, f)
                
                # After the calculation is completed, the control file parameter
                # is changed to 1 to indicate that it has completed
                with open("subprocess/sub{}.control".format(param[1]), 'wb') as f:
                    control_param[0] = 1
                    dump(control_param, f)
                
                logger.info("{} | Calculation Results Dumped | {} hours\n".format(param[1], np.round((time()-start)/3600, 4)))
            
        except Exception as exc:
            logger.critical("Error completing Calculation | {}".format(exc))
            logger.exception(exc)
            pass
        
        sleep(10)

        try:
            # when the main node has completed all of its calculations, it will
            # create a close file that triggers this code to complete
            with open('subprocess/close{}'.format(param[1]), 'r') as f:
                d = f.read()
            not_close = False
            logger.debug("{} | Close Command Found".format(param[1]))
        except FileNotFoundError:
            pass
            
    with open("subprocess/sub{}.start".format(param[1]), 'a') as f:
        f.write("subprocess finished successfully\n\n")
    logger.info("{} | Subprocess Finished".format(param[1]))