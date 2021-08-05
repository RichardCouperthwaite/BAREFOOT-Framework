# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:36:19 2021

@author: Richard Couperthwaite
"""

import numpy as np
from pyDOE import lhs
from kmedoids import kMedoids
from scipy.spatial import distance_matrix
from acquisitionFunc import expected_improvement, knowledge_gradient, thompson_sampling, upper_conf_bound, probability_improvement
from sklearn_extra.cluster import KMedoids
import pandas as pd
from pickle import load, dump
from scipy.stats import norm

# import logging
# # create logger to output framework progress
# strLog = logging.getLogger("StreamLog")
# strLog.setLevel(logging.DEBUG)
# sh = logging.StreamHandler()
# sh.setLevel(logging.DEBUG)
# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(name)s - %(message)s')
# sh.setFormatter(formatter)
# # add the handler to the logger
# strLog.addHandler(sh)

def k_medoids(sample, num_clusters):
    # clusters the samples into the number of clusters (num_clusters) according 
    # to the K-Medoids clustering algorithm and returns the medoids and the 
    # samples that belong to each cluster
    D = distance_matrix(sample, sample)
    M, C = kMedoids(D, num_clusters)
    return M, C  

def kmedoids_max(input_arr, n_clust):
    kmedoids = KMedoids(n_clusters=n_clust, random_state=0).fit(input_arr)
    new_medoids = []
    for ii in range(n_clust):
        new_cluster = input_arr[np.where(kmedoids.labels_ == ii)]
        try:
            max_index = np.where(input_arr[:,0] == np.max(new_cluster[:,0]))
            new_medoids.append(max_index[0][0])
        except ValueError:
            pass
    
    return np.array(new_medoids)

def call_model(param):
    # this function is used to call any model given in the dictionary of
    # parameters (param)
    output = param["Model"](param["Input Values"])
    return output

def cartesian(*arrays):
    # combines a set of arrays (one per dimension) so that all combinations of
    # all the arrays are in a single matrix with columns for each dimension
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape

def composition_sampler(ndim, nsamples):
    X = lhs(ndim+1, nsamples)
    X = -np.log(X)/(np.tile(np.sum(-np.log(X), axis=1), (X.shape[1],1)).transpose())
    return X[:,0:ndim]

def sampleDesignSpace(ndim, nsamples, sampleScheme):
    # This function provides three approaches to sampling of the design space
    # firstly, Latin hypercube sampling (LHS)
    # secondly, a grid based appraoch (Grid)
    # and the final approach allows for custom sampling of specific values
    # in this last approach, any additional samples required are found by 
    # Latin Hypercube sampling
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
    if sampleScheme == "CompFunc":
        x = composition_sampler(ndim, nsamples)
    return np.array(x)

def apply_constraints(samples, ndim, resolution=[], A=[], b=[], Aeq=[], beq=[], 
                      lb=[], ub=[], func=[], sampleScheme="LHS", opt_sample_size=True,
                      evaluatedPoints=[]):
    # This function handles the sampling of the design space and the application 
    # of the constraints to ensure that any points sampled satisfy the constratints
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
        
        # Duplicate Check: if a particular sample has been queried from all models
        # it needs to be removed from the potential samples. This won't stop duplicates
        # getting in since we can't exclude a point till it has been evaluated from all models
        if evaluatedPoints != []:
            all_test = np.zeros_like(constr_check)
            for evalPoints in evaluatedPoints:
                res = (x[:, None] == evalPoints).all(-1).any(-1)
                all_test += res
            all_test[np.where(all_test<len(evaluatedPoints))] = 0
            
            constr_check += all_test
            
        index = np.where(constr_check == 0)[0]
        
        # If it is chosen to optimize the sample size, the loop is continued to 
        # ensure that as close to the required number of samples are acquired
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
        # if the choice is to not optimize, the samples that pass all constraints
        # will be returned. This can lead to less samples than specified.
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
    # # Add the actual input values for the maximum of the fused model
    # if len(x_test.shape) > 1:
    #     for ii in range(x_test.shape[1]):
    #         output.append(x_test[index_max,ii])
    # else:
    #     output.append(x_test[index_max])
    # # Add the input values for the maximum knowledge gradient value
    # for i in range(x_test.shape[1]):
    #     output.append(x_test[x_star,i])
    # # Return the results
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
    fused_var = np.diag(fused_var)
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
    # # Add the actual input values for the maximum of the fused model
    # if len(x_test.shape) > 1:
    #     for ii in range(x_test.shape[1]):
    #         output.append(x_test[index_max,ii])
    # else:
    #     output.append(x_test[index_max])
    # # Add the input values for the maximum knowledge gradient value
    # for i in range(x_test.shape[1]):
    #     output.append(x_test[x_star,i])
    # # Return the results
    return output   



def calculate_TS(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Thompson Sampling of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum Thompson Sampling Result and the maximum of the fused model

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
    fused_var = np.diag(fused_var)
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
    nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # # Add the actual input values for the maximum of the fused model
    # if len(x_test.shape) > 1:
    #     for ii in range(x_test.shape[1]):
    #         output.append(x_test[index_max,ii])
    # else:
    #     output.append(x_test[index_max])
    # # Add the input values for the maximum knowledge gradient value
    # for i in range(x_test.shape[1]):
    #     output.append(x_test[x_star,i])
    # # Return the results
    return output   


def calculate_PI(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Probability of Improvement of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum Probability of Improvement and the maximum of the fused model

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
    fused_var = np.diag(fused_var)
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
    nu_star, x_star, NU = probability_improvement(curr_max, 0.01, fused_mean, np.sqrt(fused_var))
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # # Add the actual input values for the maximum of the fused model
    # if len(x_test.shape) > 1:
    #     for ii in range(x_test.shape[1]):
    #         output.append(x_test[index_max,ii])
    # else:
    #     output.append(x_test[index_max])
    # # Add the input values for the maximum knowledge gradient value
    # for i in range(x_test.shape[1]):
    #     output.append(x_test[x_star,i])
    # # Return the results
    return output   



def calculate_UCB(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Upper Confidence Bound of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum Upper Confidence Bound and the maximum of the fused model

    """
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (iteration, model_data, x_fused, fused_model_HP, \
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
    fused_var = np.diag(fused_var)
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
    
    beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
    
    kt = np.sqrt(0.2 * beta)
    
    nu_star, x_star, NU = upper_conf_bound(kt, fused_mean, np.sqrt(fused_var))
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # # Add the actual input values for the maximum of the fused model
    # if len(x_test.shape) > 1:
    #     for ii in range(x_test.shape[1]):
    #         output.append(x_test[index_max,ii])
    # else:
    #     output.append(x_test[index_max])
    # # Add the input values for the maximum knowledge gradient value
    # for i in range(x_test.shape[1]):
    #     output.append(x_test[x_star,i])
    # # Return the results
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
    (iteration, model_data, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi, sampleOpt) = data[param[0]]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)

    # strLog.critical("X-Test Shape: {}".format(x_test.shape))
    
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    
    # strLog.critical("Fused Mean Shape - {}".format(fused_mean.shape))
    
    if sampleOpt == "TS":
        """
        Thompson sampling approach
        This approach uses the uncertainty, but is quite significantly slower
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
        output = [nu_star, x_star]
    elif sampleOpt == "EI":
        """
        Expected Improvement approach
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = expected_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "PI":
        """
        Probability of Improvement approach
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = probability_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "UCB":
        """
        Upper Confidence Bound approach
        """
        beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
        kt = np.sqrt(0.2 * beta)
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = upper_conf_bound(kt, 
                                                fused_mean, 
                                                fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "KG":
        """
        Knowledge Gradient approach
        """
        nu_star, x_star, NU = knowledge_gradient(x_test.shape[0], 
                                                  0.1, 
                                                  fused_mean, 
                                                  fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "Hedge":
        output = []
        nu, x, NU = knowledge_gradient(x_test.shape[0], 
                                                  0.1, 
                                                  fused_mean, 
                                                  fused_var)
        output.append([nu, x])
        fused_var = np.diag(fused_var)
        nu, x, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
        output.append([nu, x])
        nu, x, NU = expected_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output.append([nu, x])
        nu = np.max(fused_mean)
        try:
            x = int(np.nonzero(fused_mean == nu)[0])
        except TypeError:
            x = int(np.nonzero(fused_mean == nu)[0][0])
        output.append([nu, x])
        nu, x, NU = probability_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output.append([nu, x])
        beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
        kt = np.sqrt(0.2 * beta)
        nu, x, NU = upper_conf_bound(kt, 
                                    fused_mean, 
                                    fused_var)
        output.append([nu, x])
    else:
        """
        Greedy Sampling Approach
        """
        # Find the maximum of the fused model
        nu_star = np.max(fused_mean)
        try:
            x_star = int(np.nonzero(fused_mean == nu_star)[0])
        except TypeError:
            x_star = int(np.nonzero(fused_mean == nu_star)[0][0])
        output = [nu_star, x_star]
        
    # return the maximum value and the index of the test point that corresponds
    # with the maximum value
    return output, x_test.shape[0]

def calculate_GPHedge(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        values from all acquisition functions for use in the GP Hedge portfolio
        optimization appraoch.

    Returns
    -------
    results : list
        The output from the module contains the maximum of all acquisition functions
        and the x values associated with these points.

    """
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (iteration, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
    # Initialize the output  
    output = [[0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm]]
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
    output[0][0] = np.max(fused_mean)
    output[1][0] = np.max(fused_mean)
    output[2][0] = np.max(fused_mean)
    output[3][0] = np.max(fused_mean)
    output[4][0] = np.max(fused_mean)
    output[5][0] = np.max(fused_mean)

    nu_star = []
    x_star = []
    
    
    #################
    ################
    # Need to convert this next section to run in parallel to reduce the time
    
    """
    Knowledge Gradient approach
    """
    nu_star, x_star, NU = knowledge_gradient(x_test.shape[0], 
                                              0.1, 
                                              fused_mean, 
                                              fused_var)
    output[0][1] = nu_star/cost[jj]
    output[0][2] = x_star

    """
    Thompson sampling approach
    This approach uses the uncertainty, but is quite significantly slower
    """
    fused_var = np.diag(fused_var)
    nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
    output[1][1] = nu_star/cost[jj]
    output[1][2] = x_star
    
    """
    Expected Improvement approach
    """
    nu_star, x_star, NU = expected_improvement(curr_max, 
                                        0.01, 
                                        fused_mean, 
                                        fused_var)
    output[2][1] = nu_star/cost[jj]
    output[2][2] = x_star
   
    """
    Greedy Sampling Approach
    """
    # Find the maximum of the fused model
    nu_star = np.max(fused_mean)
    try:
        x_star = int(np.nonzero(fused_mean == nu_star)[0])
    except TypeError:
        x_star = int(np.nonzero(fused_mean == nu_star)[0][0])
    output[3][1] = nu_star/cost[jj]
    output[3][2] = x_star
    
    """
    Probability of Improvement Approach
    """
    # Find the maximum of the fused model
    nu_star, x_star, NU = probability_improvement(curr_max, 
                                        0.01, 
                                        fused_mean, 
                                        fused_var)
    output[4][1] = nu_star/cost[jj]
    output[4][2] = x_star
    
    """
    Upper Confidence Bound Approach
    """
    beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
    
    kt = np.sqrt(0.2 * beta)
    # Find the maximum of the fused model
    nu_star, x_star, NU = upper_conf_bound(kt, 
                                        fused_mean, 
                                        fused_var)
    output[5][1] = nu_star/cost[jj]
    output[5][2] = x_star
    
    
    
    # # Add the actual input values for the maximum of the fused model
    # if len(x_test.shape) > 1:
    #     for ii in range(x_test.shape[1]):
    #         output[0].append(x_test[index_max,ii])
    #         output[1].append(x_test[index_max,ii])
    #         output[2].append(x_test[index_max,ii])
    #         output[3].append(x_test[index_max,ii])
    #         output[4].append(x_test[index_max,ii])
    #         output[5].append(x_test[index_max,ii])
    # else:
    #     output[0].append(x_test[index_max])
    #     output[1].append(x_test[index_max])
    #     output[2].append(x_test[index_max])
    #     output[3].append(x_test[index_max])
    #     output[4].append(x_test[index_max])
    #     output[5].append(x_test[index_max])
        
    # for i in range(x_test.shape[1]):
    #     output[0].append(x_test[output[0][2],i])
    #     output[1].append(x_test[output[1][2],i])
    #     output[2].append(x_test[output[2][2],i])
    #     output[3].append(x_test[output[3][2],i])
    #     output[4].append(x_test[output[4][2],i])
    #     output[5].append(x_test[output[5][2],i])
        
    return output

def calculate_Greedy(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Maximum of a fused model constructed out of a reification 
        model object for Greedy optimization

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to the maximum of the fused model

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
    fused_var = np.diag(fused_var)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    if index_max_[0].shape[0] > 1:
        index_max = int(index_max_[0][0])
    else:
        index_max = int(index_max_[0])
    # try:
    #     index_max = int(index_max_)
    # except TypeError:
    #     try:
    #         index_max = int(index_max_[0])
    #     except TypeError:
    #         index_max = int(index_max_[0][0])
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = np.max(fused_mean)
    output[2] = index_max
    # Add the actual input values for the maximum of the fused model
    for kk in range(2):
        if len(x_test.shape) > 1:
            for ii in range(x_test.shape[1]):
                output.append(x_test[index_max,ii])
        else:
            output.append(x_test[index_max])
    # Return the results
    return output   


def evaluateFusedModel(param):
    # in order to update the gains for the GP Hedge Portfolio optimization scheme
    # it is necessary to query the next best points predicted by all the acquisition
    # functions.
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, reification, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi, acqIndex) = data[param[0]]
    if reification:
        # Create the fused model
        model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                    fused_model_HP[0], 0.1, 
                                    kernel)
        fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    else:
        model_temp.l_param = fused_model_HP[1:]
        model_temp.sigma_f = fused_model_HP[0]
        model_temp.kk = model_temp.create_kernel()
        model_temp.gp = model_temp.create_gp()
        fused_mean, fused_var = model_temp.predict_cov(x_test)
    return [acqIndex, fused_mean]

def batchAcquisitionFunc(param):
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
        
    with open("data/reificationObj", 'rb') as f:
        modelGP = load(f)

    xi = 0.01
    iteration, x_test, fusedModelHP, curr_max, acqFunc, extra = data[param[0]]
    
    if acqFunc == "EI-BMARS":
        pass
    else:
        if type(modelGP) == list:
            modelGP[0].l_param = fusedModelHP[1:]
            modelGP[0].sigma_f = fusedModelHP[0]
            modelGP[0].kk = modelGP[0].create_kernel()
            modelGP[0].gp = modelGP[0].create_gp()
            modelGP[1].l_param = fusedModelHP[1:]
            modelGP[1].sigma_f = fusedModelHP[0]
            modelGP[1].kk = modelGP[1].create_kernel()
            modelGP[1].gp = modelGP[1].create_gp()
            pareto, goal, ref = extra
            means = np.zeros((x_test.shape[0], 2))
            sigmas = np.zeros((x_test.shape[0], 2))
            
            m1, v1 = modelGP[0].predict_cov(x_test)
            m2, v2 = modelGP[1].predict_cov(x_test)
            
            means[:,0] = m1
            means[:,1] = m2
            sigmas[:,0] = np.sqrt(np.diag(v1))
            sigmas[:,1] = np.sqrt(np.diag(v2))
            
            N_obj = 2 ## number of objectives
        else:
            modelGP.l_param = fusedModelHP[1:]
            modelGP.sigma_f = fusedModelHP[0]
            modelGP.kk = modelGP.create_kernel()
            modelGP.gp = modelGP.create_gp()
            fused_mean, fused_var = modelGP.predict_cov(x_test)
    
    if acqFunc == "EHVI":
        # ## Turn the problem into minimizing for all objectives:
        # ### this is essential as the method works for minimizing
        for i in range(goal.shape[0]):
            means[:,i]=-1*means[:,i]
            pareto[:,i]=-1*pareto[:,i]
    
        
        
        ## Sorting the non_dominated points considering the first objective
        ##### It does not matter which objective to sort but lets do it with the
        ##### 1st objective
        I = np.argsort(pareto[:, 0])
        pareto = pareto[I,:]
        
        ## Finding useless test points
        ### this is done by checking if one is dominated with 95# certainty. (2 sigma)
        ### so that if a test points has a very small probability to improve the
        ### hypervolume, we discard it to avoid unnecessary EHVI calculations
        
        temp = means-2.*sigmas;
        
        ind = np.zeros((means.shape[0],1))
        ehvi = np.zeros((means.shape[0],1))
        
        for i in range(means.shape[0]):
            diff=pareto-temp[i,:]
            for j in range(diff.shape[0]):
                if np.max(diff[j,:])<0:
                    ind[i,0]=1;
        
        ## EHVI calculation for test points
        for i in range(means.shape[0]):
            if ind[i]==1:
                ehvi[i,0]=0
            else:
                hvi = 0
                box = 1
                ### EHVI over the box from infinity to the ref point
                for j in range(N_obj):
                    s = (ref[j]-means[i,j])/sigmas[i,j]
                    box = box*((ref[j]-means[i,j])*norm.cdf(s)+sigmas[i,j]*norm.pdf(s));
    
                ### calculate how much adding a test point can improve the hypervolume
                #         hvi = recursive(means(i,:),sigmas(i,:),ref,pareto);
                
                for zz in range(pareto.shape[0]-1):                
                    a = pareto[zz,:]
                    aa = np.maximum(pareto[zz,:],pareto[zz+1])
                    hvi_temp1=1
                    hvi_temp2=1
                    
                    for j in range(N_obj):
                        s_up = (ref[j]-means[j])/sigmas[j]
                        s_low = (a[j]-means[j])/sigmas[j]
                        up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up))
                        low = ((a[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low))
                        hvi_temp1 = hvi_temp1 * (up-low);
    
                        s_up = (ref[j]-means[j])/sigmas[j]
                        s_low = (aa[j]-means[j])/sigmas[j]
                        up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up))
                        low = ((aa[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low))
                        hvi_temp2 = hvi_temp2 * (up-low)
                    
                    
                    hvi = hvi + hvi_temp1 - hvi_temp2;
                
                a=pareto[-1,:]
                hvi_temp1=1
                for j in range(N_obj):
                    s_up = (ref[j]-means[j])/sigmas[j];
                    s_low = (a[j]-means[j])/sigmas[j];
                    up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up));
                    low = ((a[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low));
                    hvi_temp1 = hvi_temp1 * (up-low);
                hvi = hvi + hvi_temp1;
                ehvi[i,0]=box-(hvi[0])
        nu_star = np.max(ehvi)
        x_star = np.where(ehvi == np.max(ehvi))[0][0]
        output = [nu_star, x_star]
    elif acqFunc == "TS":
        """
        Thompson sampling approach
        This approach uses the uncertainty, but is quite significantly slower
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
        output = [nu_star, x_star]
    elif acqFunc == "EI":
        """
        Expected Improvement approach
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = expected_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output = [nu_star, x_star]
    elif acqFunc == "PI":
        """
        Probability of Improvement approach
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = probability_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output = [nu_star, x_star]
    elif acqFunc == "UCB":
        """
        Upper Confidence Bound approach
        """
        beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
        kt = np.sqrt(0.2 * beta)
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = upper_conf_bound(kt, 
                                                fused_mean, 
                                                fused_var)
        output = [nu_star, x_star]
    elif acqFunc == "KG":
        """
        Knowledge Gradient approach
        """
        nu_star, x_star, NU = knowledge_gradient(x_test.shape[0], 
                                                  0.1, 
                                                  fused_mean, 
                                                  fused_var)
        output = [nu_star, x_star]
    elif acqFunc == "Hedge":
        output = []
        nu, x, NU = knowledge_gradient(x_test.shape[0], 
                                                  0.1, 
                                                  fused_mean, 
                                                  fused_var)
        output.append([nu, x])
        fused_var = np.diag(fused_var)
        nu, x, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
        output.append([nu, x])
        nu, x, NU = expected_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output.append([nu, x])
        nu = np.max(fused_mean)
        try:
            x = int(np.nonzero(fused_mean == nu)[0])
        except TypeError:
            x = int(np.nonzero(fused_mean == nu)[0][0])
        output.append([nu, x])
        nu, x, NU = probability_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output.append([nu, x])
        beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
        kt = np.sqrt(0.2 * beta)
        nu, x, NU = upper_conf_bound(kt, 
                                    fused_mean, 
                                    fused_var)
        output.append([nu, x])
    elif acqFunc == "EI-BMARS":
        pass
    else:
        """
        Greedy Sampling Approach
        """
        # Find the maximum of the fused model
        nu_star = np.max(fused_mean)
        try:
            x_star = int(np.nonzero(fused_mean == nu_star)[0])
        except TypeError:
            x_star = int(np.nonzero(fused_mean == nu_star)[0][0])
        output = [nu_star, x_star]
        
    # return the maximum value and the index of the test point that corresponds
    # with the maximum value    
    return output

def Pareto_finder(V,goal):
    
    # V is a matrix, each row is the objectives of one design points
    
    # goal : a row vector to define which objectives to be minimized or
    # maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]
    
    
    # Turn the problem into minimizing for all objectives:
    for i in range(goal.shape[0]):
    # for i = 1 : size(goal,2)
        # if goal[i]==1:
        V[:,i]=-1*V[:,i]

    pareto=[]
    ind=[]
    
    for i in range(V.shape[0]):
        p = V[i,:]
        s = np.delete(V,i,0)
        # s[i,:]=[]
        trig = 0
        for j in range(s.shape[0]):
            temp = p-s[j,:]
            if np.min(temp)>=0:
                trig=1 # this means vector p is dominated

        if trig==0:
            pareto.append(p)
            ind.append(i)
    
    pareto = np.array(pareto)
    if len(pareto.shape) == 1:
        pareto = np.expand_dims(pareto, axis=0)
    ind = np.array(ind)
    # Changing back the signs if were changed before.
    for i in range(goal.shape[0]):
        # if goal[i]==1:
        pareto[:,i]=-1*pareto[:,i]

    return pareto, ind

def calculate_EHVI(param):
    # means : GP mean estimation of objectives of the test points (fused means in
    # multifidelity cases). Each column for 1 objective values
    
    # sigmas : uncertainty of GP mean estimations (std). Each column for 1 objective
    
    # goal : a row vector to define which objectives to be minimized or
    # maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]
    
    # ref : hypervolume reference for calculations
    
    # pareto : Current true pareto front obtained so far
    
    ########### Note that in all variables, the order of columns should be the
    ########### same. For example, the 1st column of all matrices above is
    ########### related to the objective 1. Basically, each row = 1 design
    ##########################################################################
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
        
    mean_train, goal, ref, pareto = model_data
    
    # Initialize the output  
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp[0].update_GP(np.expand_dims(x_test[kk], axis=0), mean_train[0], jj)
    model_temp[0].create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    model_temp[1].update_GP(np.expand_dims(x_test[kk], axis=0), mean_train[1], jj)
    model_temp[1].create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    
    means = np.zeros((x_test.shape[0], 2))
    sigmas = np.zeros((x_test.shape[0], 2))
    
    m1, v1 = model_temp[0].predict_fused_GP(x_test)
    m2, v2 = model_temp[1].predict_fused_GP(x_test)
    
    means[:,0] = m1
    means[:,1] = m2
    sigmas[:,0] = np.sqrt(np.diag(v1))
    sigmas[:,1] = np.sqrt(np.diag(v2))
    
    N_obj = 2 ## number of objectives
    
    
    # ## Turn the problem into minimizing for all objectives:
    # ### this is essential as the method works for minimizing
    for i in range(goal.shape[0]):
        means[:,i]=-1*means[:,i]
        pareto[:,i]=-1*pareto[:,i]
    
    ## Sorting the non_dominated points considering the first objective
    ##### It does not matter which objective to sort but lets do it with the
    ##### 1st objective
    I = np.argsort(pareto[:, 0])
    pareto = pareto[I,:]
    
    ## Finding useless test points
    ### this is done by checking if one is dominated with 95# certainty. (2 sigma)
    ### so that if a test points has a very small probability to improve the
    ### hypervolume, we discard it to avoid unnecessary EHVI calculations
    
    temp = means-2.*sigmas;
    
    ind = np.zeros((means.shape[0],1))
    ehvi = np.zeros((means.shape[0],1))
    
    for i in range(means.shape[0]):
        diff=pareto-temp[i,:]
        for j in range(diff.shape[0]):
            if np.max(diff[j,:])<0:
                ind[i,0]=1;
    
    ## EHVI calculation for test points
    for i in range(means.shape[0]):
        if ind[i]==1:
            ehvi[i,0]=0
        else:
            hvi = 0
            box = 1
            ### EHVI over the box from infinity to the ref point
            for j in range(N_obj):
                s = (ref[j]-means[i,j])/sigmas[i,j]
                box = box*((ref[j]-means[i,j])*norm.cdf(s)+sigmas[i,j]*norm.pdf(s));

            ### calculate how much adding a test point can improve the hypervolume
            #         hvi = recursive(means(i,:),sigmas(i,:),ref,pareto);
            
            for zz in range(pareto.shape[0]-1):                
                a = pareto[zz,:]
                aa = np.maximum(pareto[zz,:],pareto[zz+1])
                hvi_temp1=1
                hvi_temp2=1
                
                for j in range(N_obj):
                    s_up = (ref[j]-means[j])/sigmas[j]
                    s_low = (a[j]-means[j])/sigmas[j]
                    up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up))
                    low = ((a[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low))
                    hvi_temp1 = hvi_temp1 * (up-low);

                    s_up = (ref[j]-means[j])/sigmas[j]
                    s_low = (aa[j]-means[j])/sigmas[j]
                    up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up))
                    low = ((aa[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low))
                    hvi_temp2 = hvi_temp2 * (up-low)
                
                
                hvi = hvi + hvi_temp1 - hvi_temp2;
            
            a=pareto[-1,:]
            hvi_temp1=1
            for j in range(N_obj):
                s_up = (ref[j]-means[j])/sigmas[j];
                s_low = (a[j]-means[j])/sigmas[j];
                up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up));
                low = ((a[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low));
                hvi_temp1 = hvi_temp1 * (up-low);
            hvi = hvi + hvi_temp1;
            ehvi[i,0]=box-(hvi[0])

    output[1] = np.max(ehvi)
    output[2] = np.where(ehvi == np.max(ehvi))[0][0]
    return output

def fused_EHVI(param):
    # means : GP mean estimation of objectives of the test points (fused means in
    # multifidelity cases). Each column for 1 objective values
    
    # sigmas : uncertainty of GP mean estimations (std). Each column for 1 objective
    
    # goal : a row vector to define which objectives to be minimized or
    # maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]
    
    # ref : hypervolume reference for calculations
    
    # pareto : Current true pareto front obtained so far
    
    ########### Note that in all variables, the order of columns should be the
    ########### same. For example, the 1st column of all matrices above is
    ########### related to the objective 1. Basically, each row = 1 design
    ##########################################################################
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (iteration, model_data, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi, sampleOpt) = data[param[0]]
    # Create the fused model
    model_temp[0].create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    model_temp[1].create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    
    # Initialize the output  
    pareto, goal, ref = model_data
    output = [0,0]
        
    means = np.zeros((x_test.shape[0], 2))
    sigmas = np.zeros((x_test.shape[0], 2))
    
    m1, v1 = model_temp[0].predict_fused_GP(x_test)
    m2, v2 = model_temp[1].predict_fused_GP(x_test)
    
    means[:,0] = m1
    means[:,1] = m2
    sigmas[:,0] = np.sqrt(np.diag(v1))
    sigmas[:,1] = np.sqrt(np.diag(v2))
    
    N_obj = 2 ## number of objectives
    
    
    # ## Turn the problem into minimizing for all objectives:
    # ### this is essential as the method works for minimizing
    for i in range(goal.shape[0]):
        means[:,i]=-1*means[:,i]
        pareto[:,i]=-1*pareto[:,i]

    
    ## Sorting the non_dominated points considering the first objective
    ##### It does not matter which objective to sort but lets do it with the
    ##### 1st objective
    I = np.argsort(pareto[:, 0])
    pareto = pareto[I,:]
    
    ## Finding useless test points
    ### this is done by checking if one is dominated with 95# certainty. (2 sigma)
    ### so that if a test points has a very small probability to improve the
    ### hypervolume, we discard it to avoid unnecessary EHVI calculations
    
    temp = means-2.*sigmas;
    
    ind = np.zeros((means.shape[0],1))
    ehvi = np.zeros((means.shape[0],1))
    
    for i in range(means.shape[0]):
        diff=pareto-temp[i,:]
        for j in range(diff.shape[0]):
            if np.max(diff[j,:])<0:
                ind[i,0]=1;
    
    ## EHVI calculation for test points
    for i in range(means.shape[0]):
        if ind[i]==1:
            ehvi[i,0]=0
        else:
            hvi = 0
            box = 1
            ### EHVI over the box from infinity to the ref point
            for j in range(N_obj):
                s = (ref[j]-means[i,j])/sigmas[i,j]
                box = box*((ref[j]-means[i,j])*norm.cdf(s)+sigmas[i,j]*norm.pdf(s));

            ### calculate how much adding a test point can improve the hypervolume
            #         hvi = recursive(means(i,:),sigmas(i,:),ref,pareto);
            
            for zz in range(pareto.shape[0]-1):                
                a = pareto[zz,:]
                aa = np.maximum(pareto[zz,:],pareto[zz+1])
                hvi_temp1=1
                hvi_temp2=1
                
                for j in range(N_obj):
                    s_up = (ref[j]-means[j])/sigmas[j]
                    s_low = (a[j]-means[j])/sigmas[j]
                    up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up))
                    low = ((a[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low))
                    hvi_temp1 = hvi_temp1 * (up-low);

                    s_up = (ref[j]-means[j])/sigmas[j]
                    s_low = (aa[j]-means[j])/sigmas[j]
                    up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up))
                    low = ((aa[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low))
                    hvi_temp2 = hvi_temp2 * (up-low)
                
                
                hvi = hvi + hvi_temp1 - hvi_temp2;
            
            a=pareto[-1,:]
            hvi_temp1=1
            for j in range(N_obj):
                s_up = (ref[j]-means[j])/sigmas[j];
                s_low = (a[j]-means[j])/sigmas[j];
                up = ((ref[j]-means[j])*norm.cdf(s_up)+sigmas[j]*norm.pdf(s_up));
                low = ((a[j]-means[j])*norm.cdf(s_low)+sigmas[j]*norm.pdf(s_low));
                hvi_temp1 = hvi_temp1 * (up-low);
            hvi = hvi + hvi_temp1;
            ehvi[i,0]=box-(hvi[0])

    output[0] = np.max(ehvi)
    output[1] = np.where(ehvi == np.max(ehvi))[0][0]
    return output, x_test.shape[0]

def storeObject(obj, filename):
    with open(filename, 'wb') as f:
        dump(obj, f)
        
        
def checkInitInputs(*args, **kwargs):
    pass

def checkParameterInputs(*args, **kwargs):
    pass
