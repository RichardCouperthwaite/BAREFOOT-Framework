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
from util import calculate_KG, calculate_EI, fused_calculate, calculate_TS
import logging

if __name__ == "__main__":
    param = argv
    
    # log_level = logging.DEBUG
    log_level = logging.INFO
    
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
        
    with open("subprocess/sub{}.start".format(param[1]), 'w') as f:
        f.write("subprocess started successfully\n\n") 
    
    not_close = True

    while not_close:
        try:
            try:
                with open("subprocess/sub{}.control".format(param[1]), 'rb') as f:
                    control_param = load(f)
                with open("subprocess/sub{}.start".format(param[1]), 'a') as f:
                    f.write("Control File Found - {} | {}\n".format(control_param[0], control_param[1]))
                    logger.debug("Control File Found - {} | {}\n".format(control_param[0], control_param[1]))
            except FileExistsError:
                logger.debug("Control File could not be found")
                control_param = [1,1]
 
            if control_param[0] == 0:
                logger.info("{} | New Subprocess calculation started\n".format(param[1]))

                if control_param[2] == "KG":
                    function = calculate_KG
                elif control_param[2] == "EI":
                    function = calculate_EI
                elif control_param[2] == "TS":
                    function = calculate_TS
                
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
                            if count % 200 == 0:
                                logger.info("{} | {} / {} Calculations Completed".format(param[1], count, len(parameters)))
                            
                    with open("subprocess/{}.output".format(param[1]), 'wb') as f:
                        dump(kg_output, f)
                    
                    
                elif control_param[1] == "fused":
                    with open("subprocess/{}.dump".format(param[1]), 'rb') as f:
                        parameters = load(f)
                    logger.debug("{} | Fused Model Calculation Started | {} Calculations".format(param[1], len(parameters)))
                    fused_output = []
                    count = 0
                    with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
                        for result_from_process in zip(parameters, executor.map(fused_calculate,parameters)):
                            params, results = result_from_process
                            fused_output.append(results[0])
                            count += 1
                            if count % 200 == 0:
                                logger.info("{} | {} / {} Calculations Completed".format(param[1], count, len(parameters)))

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
            
                with open("subprocess/sub{}.control".format(param[1]), 'wb') as f:
                    control_param[0] = 1
                    dump(control_param, f)
                
                logger.info("{} | Calculation Results Dumped | {} hours\n".format(param[1], np.round((time()-start)/3600, 4)))
            
        except Exception as exc:
            logger.critical("Error completing Calculation | {}".format(exc))
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