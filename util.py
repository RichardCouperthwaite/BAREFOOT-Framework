# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:36:19 2021

@author: Richard Couperthwaite
"""

import numpy as np
from pyDOE import lhs
from kmedoids import kMedoids
from scipy.spatial import distance_matrix
from acquisitionFunc import expected_improvement, knowledge_gradient, thompson_sampling
import pandas as pd
from pickle import load

# from testFrameworkCodeUtil import ThreeHumpCamel, ThreeHumpCamel_LO1, ThreeHumpCamel_LO2, ThreeHumpCamel_LO3, plotResults 
# from testFrameworkCodeUtil import isostress_IS, isostrain_IS, isowork_IS, EC_Mart_IS, secant1_IS, TC_GP, RVE_GP

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

def sampleDesignSpace(ndim, nsamples, sampleScheme):
    if sampleScheme == "LHS":
        x = lhs(ndim, nsamples)
    if sampleScheme == "Grid":
        for jjj in range(nsamples-1):
            input_arr = np.linspace(0,1,jjj+1)
            all_arr = []
            for ii in range(ndim):
                all_arr.append(input_arr)
            x = cartesian(*all_arr)
            if x.shape[0] >= nsamples:
                return x
    if sampleScheme == "Custom":
        dfInputs = pd.read_csv("data/possibleInputs.csv", index_col=0)
        if dfInputs.shape[0] > nsamples:
            x = dfInputs.sample(n=nsamples)
        else:
            x_other = pd.DataFrame(lhs(ndim, nsamples-dfInputs.shape[0]),columns=dfInputs.columns)
            x = pd.concat((dfInputs, x_other))         
    return np.array(x)

def apply_constraints(samples, ndim, resolution=[], A=[], b=[], Aeq=[], beq=[], lb=[], ub=[], func=[], sampleScheme="LHS", opt_sample_size=True):
    sampleSelection = True
    constraints = np.zeros((5))
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
    
    x_largest = []
    largest_set = 0
    
    while sampleSelection:
        try:
            x = sampleDesignSpace(ndim, lhs_samples, sampleScheme)
        except:
            x = sampleDesignSpace(ndim, lhs_samples, "LHS")
        if resolution != []:
            x = np.round(x, decimals=resolution)
        constr_check = np.zeros((x.shape[0], ndim))
        
        # Apply inequality constraints
        if (A != []) and (b != []) and (len(A) == ndim):
            A_tile = np.tile(np.array(A), (x.shape[0],1))
            constr_check += A_tile*x <= b
            constraints[0] = 0

        # Apply equality constraints
        if (Aeq != []) and (beq != []):
            Aeq_tile = np.tile(np.array(Aeq), (x.shape[0],1))
            constr_check += Aeq_tile*x <= beq
            constraints[1] = 0
        
        # Apply Lower and Upper Bounds
        if (lb != []) and (len(lb) == ndim):
            lb_tile = np.tile(np.array(lb).reshape((1,ndim)), (x.shape[0],1))
            constr_check += x < lb_tile
            constraints[2] = 0
        if (ub != []) and (len(ub) == ndim):
            ub_tile = np.tile(np.array(ub).reshape((1,ndim)), (x.shape[0],1))
            constr_check += x > ub_tile
            constraints[3] = 0
        
        constr_check = np.sum(constr_check, axis=1)
        
        # Apply custom function constraints
        if (type(func) == list) and (func != []):
            for ii in range(len(func)):
                try:
                    constr_check += func[ii](x)
                    constraints[4] -= 1
                except:
                    pass
        elif (type(func) != list) and (func != []):
            try:
                constr_check += func(x)    
                constraints[4] = 0
            except:
                pass
        index = np.where(constr_check == 0)[0]
        
        if opt_sample_size:
            if index.shape[0] >= samples:
                x = x[index[0:samples],:]
                sampleSelection = False
                if np.sum(constraints) != 0:
                    const_satisfied = False
                else:
                    const_satisfied = True
            else:
                if len(index) > largest_set:
                    largest_set = len(index)
                    x_largest = x[index,:]
                if lhs_samples/samples < ndim*2000:
                    lhs_samples += samples*100
                else:
                    x = x_largest
                    sampleSelection = False
                    const_satisfied = False
        else:
            x = x[index,:]
            sampleSelection = False
            const_satisfied = True
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
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.update_GP(*model_data)
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
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
    # Initialize the output  
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.update_GP(*model_data)
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



def calculate_TS(param):
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
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
    # Initialize the output  
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.update_GP(*model_data)
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
    nu_star, x_star, NU = thompson_sampling(x_test, model_temp.fused_GP)
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
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi) = data[param[0]]
    print("starting TS Calc | {}".format(len(x_test)))
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    """
    Greedy Sampling Approach
    """
    # Predict the mean and variance at each test point
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    # Find the maximum of the fused model
    nu_star = np.max(fused_mean)
    try:
        x_star = int(np.nonzero(fused_mean == nu_star)[0])
    except TypeError:
        x_star = int(np.nonzero(fused_mean == nu_star)[0][0])
    """
    Thompson sampling approach
    This approach uses the uncertainty, but is quite significantly slower
    """
    # nu_star, x_star, NU = thompson_sampling(x_test, model_temp.fused_GP)
    
    # return the maximum value and the index of the test point that corresponds
    # with the maximum value
    return [nu_star,x_star], x_test.shape[0]

if __name__ == "__main__":
    x, check = apply_constraints(100, 7, resolution=3, sampleScheme="Custom")