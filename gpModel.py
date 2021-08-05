# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:43:08 2020

@author: richardcouperthwaite
"""

from george import kernels, GP
import numpy as np
from copy import deepcopy

class gp_model:
    """
    A class that creates a GP from a given set of input data and hyper-parameters.
    The Kernel can be selected from three separate Kernels.
    """
    def __init__(self, x_train, y_train, l_param, sigma_f, sigma_n, n_dim, kern, mean=0):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.l_param = np.array(l_param)**2
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.mean = mean
        self.n_dim = n_dim
        self.kern = kern
        self.kk = self.create_kernel()
        self.gp = self.create_gp()
        
    def create_kernel(self):
        # This function creates the covariance function kernel for the Gaussian Process
        if self.kern == 'SE':
            return self.sigma_f * kernels.ExpSquaredKernel(self.l_param, ndim=self.n_dim)
        elif self.kern == 'M32':
            return self.sigma_f * kernels.Matern32Kernel(self.l_param, ndim=self.n_dim)
        elif self.kern == 'M52':
            return self.sigma_f * kernels.Matern52Kernel(self.l_param, ndim=self.n_dim)
    
    def create_gp(self):
        # This function uses the kernel defined above to compute and train the Gaussian Process model
        gp = GP(kernel=self.kk, mean=self.mean)
        gp.compute(self.x_train, self.sigma_n)
        return gp
    
    def predict_cov(self, x_pred):
        # This function is used to predict the mean and the full covariance 
        # matrix for the test points (x_pred)
        mean, sigma = self.gp.predict(self.y_train, x_pred, kernel = self.kk, return_cov=True, return_var=False)
        return mean, sigma
    
    def predict_var(self, x_pred):
        # This function is used to predict the mean and the variance (the diagonal of 
        # the full covariance matrix) for the test points (x_pred)
        mean, var = self.gp.predict(self.y_train, x_pred, kernel = self.kk, return_cov=False, return_var=True)
        return mean, var
    
    def update(self, new_x_data, new_y_data, new_y_err, err_per_point):
        # This function is used to update and retrain the GP model when new
        # training data is available
        self.x_train = np.vstack((self.x_train, new_x_data))
        self.y_train = np.append(self.y_train, new_y_data)
        if err_per_point:
            self.sigma_n = np.append(self.sigma_n, new_y_err)
            
        self.gp = self.create_gp()
    
    def sample_posterior(self, x_test):
        # This function provides a random sampling from the Gaussian Process
        # posterior distribution
        return self.gp.sample_conditional(self.y_train, x_test, size=1)
        
    def log_likelihood(self):
        # This function computes the log likelihood of the training data given
        # the hyperparameters
        return self.gp.log_likelihood(self.y_train, quiet=True)
    
    def get_hyper_params(self):
        # This function obtains the hyperparameters from the trained GP and
        # modifies them to be consistent with other Gaussian Process implementations
        curr_params = self.gp.get_parameter_vector()
        params = []
        for i in range(len(curr_params)):
            if i == 0:
                params.append(np.exp(curr_params[i])*self.n_dim)
            else:
                params.append(np.sqrt(np.exp(curr_params[i])))
        return np.array(params)
        
    def hp_optimize(self, meth="L-BFGS-B", update=False):
        # This function can be used ot optimize the GP hyperparameters
        import scipy.optimize as op
        gp = deepcopy(self)
        p0 = gp.gp.get_parameter_vector()
        def nll(p):
            gp.gp.set_parameter_vector(p)
            ll = gp.log_likelihood()
            return -ll if np.isfinite(ll) else 1e25
        
        def grad_nll(p):
            gp.gp.set_parameter_vector(p)
            return -gp.gp.grad_log_likelihood(self.y_train, quiet=True)
                
        results = op.minimize(nll, p0, jac=grad_nll, method=meth)
        if update:
            # automatically update the hyper-parameters, the required input for
            # the set_parameter_vector command is the log of the hyper-parameters
            self.gp.set_parameter_vector(results.x)
            self.gp.compute(self.x_train, self.sigma_n)
        # The results are the log of the hyper-parameters, so return the
        # exponential of the results.
        return np.exp(results.x)
    
    



class bmarsModel():
    """
    %   This is a Bayesian MARS model for Gaussian response data:
    %	see chapters 3, 4 in "Bayesian methods for nonlinear classification and regression".
    %	(2002). Denison, Holmes, Mallick and Smith: published by Wiley. 
    %
    %	USAGE:
    %		the program simulates a Bayesian MARS model assuming a Gaussian reponse variable 
    %		using Markov chain Monte Carlo.
    """
    def __init__(self,data_in,data_output,interaction=np.array((0, 1)), order=np.array((0, 1)), k_max = 500, 
                     mcmc_samples = 1000, burn_in = 1000, thin = 10, alpha_1=0.1, alpha_2=0.1, SAVE_SAMPLES = 1):
        data = np.zeros((data_in.shape[0], data_in.shape[1]+1))
        data[:,0]  = data_output
        data[:,1:] = data_in
        self.data = data
        self.interaction = interaction
        self.order = order
        self.k_max = k_max
        self.mcmc_samples = mcmc_samples
        self.burn_in = burn_in
        self.thin = thin
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.SAVE_SAMPLES = SAVE_SAMPLES
        
    def calculate(self, test):
        out = bayes_mars_gauss(self.data, test, 1, self.interaction, self.order, self.k_max, 
                     self.mcmc_samples, self.burn_in, self.thin, self.alpha_1, self.alpha_2, self.SAVE_SAMPLES)
        self.test_set_predictions_credibles = out[0]
        self.test_set_predictions_pred_store = out[1] 
        self.chain_stats_k_store = out[2]
        self.chain_stats_LL_store = out[3]
        self.a_seq = out[4]
        self.basis_parameters_seq = out[5]
        
        return [self.test_set_predictions_credibles, self.test_set_predictions_pred_store]
    
    def update(self, new_x_data, new_y_data, new_y_err, err_per_point):
        pass





# main function for Bayesian Mars model
def bayes_mars_gauss(data, test, STANDARDISE=1, interaction=np.array((0, 1)), order=np.array((0, 1)), k_max = 500, 
                     mcmc_samples = 1000, burn_in = 1000, thin = 10, alpha_1=0.1, alpha_2=0.1, SAVE_SAMPLES = 1):

    """
%   This is a Bayesian MARS model for Gaussian response data:
%	see chapters 3, 4 in "Bayesian methods for nonlinear classification and regression".
%	(2002). Denison, Holmes, Mallick and Smith: published by Wiley. 
%
%	USAGE:
%		the program simulates a Bayesian MARS model assuming a Gaussian reponse variable 
%		using Markov chain Monte Carlo.
%
    Parameters
    ----------
    data : array
        training data: first column is Y (the response) remaining columns are covariates, X
    test : array
        test data: same format as 'data'.

    """
    
    import numpy as np
    import random
    import scipy.linalg
    from scipy.stats import invgauss
    import os
    import pickle
    import pandas as pd  
    
    # now extract information from program inputs.....
    
    # response should be stored in the first column of the data and test
    Y = data[:, 0]
    Yt = test[:, 0]
    
    # extract predictor variables
    X = data[:, 1:]
    Xt = test[:, 1:]
    
    # get dimensions of data 
    n = data.shape[0]
    nt = test.shape[0]
    p = X.shape[1]
    
    # get number of allowable interactions
    n_inter = len(interaction)
    # get number of allowable orders of the basis functions
    n_order = len(order)

    mx = np.ones(p)
    sx = np.ones(p)    
    if STANDARDISE:
        # then standardise data
        for i in range(p):
            mx[i] = np.mean(X[:, i])
            sx[i] = np.std(X[:, i])
            if sx[i] != 0:
                X[:, i] = (X[:, i] - mx[i]) / sx[i]
                Xt[:, i] = (Xt[:, i] - mx[i]) / sx[i]        
    
 
    # we will start the MCMC chain using a model with just an intercept (constant) term
    k = 1 # only one basis function in the model (the intercept)
    # allocate space for the design matrix
    X_mars = np.zeros((n, k_max))
    # set first column to intercept
    X_mars[:, 0] = 1
    # and same for the design matrix of the test data
    Xt_mars = np.zeros((nt, k_max))
    Xt_mars[:, 0] = 1

    prec = 10 # this is the precision of the coefficient prior: will be updated as part of the program
    sig2 = 1 # this is the noise variance: updated during the mcmc run.
    
    # we need Y'*Y repeatedly so lets store it
    YtY = Y.T.dot(Y)
    
    # get log marginal likelihood of current intercept model and a draw of the basis coefficients, beta, 
    # the mean of beta, and the posterior sum of squares alpha_star
    marg_lik, beta, beta_mean, alpha_star = get_ml(X_mars[:, :k], Y, YtY, sig2, alpha_1, alpha_2, prec)
    
    # define a class to store the basis function parameters
    class basis_para:
        def __init__(self, order, inter, knot, var, lr, mx, sx):
            self.order = order
            self.inter = inter
            self.knot = knot
            self.var = var 
            self.lr = lr
            self.mx = mx
            self.sx = sx              
    
    basis_parameters = [[]]
    basis_parameters_seq = []
    if SAVE_SAMPLES:
        # set basis function parameters for intercept:
        # these are just dummy values for the intercept but will take 
        # realistic values for each MARS basis function
        basis = basis_para(0, 0, np.zeros((1, n_inter)), np.zeros((1, n_inter)), np.zeros((1, n_inter)), 0, 0)
        basis_parameters[0] = basis
    
    # we wish to store these for output 
    cred_n = np.floor(mcmc_samples/thin)
    chain_stats_LL_store = np.zeros(int(cred_n)) # the log marginal likelihood
    chain_stats_k_store = np.zeros(int(cred_n)) # the number of basis functions
    pred_store = np.zeros(nt) # the predictions on test
    test_set_predictions_pred_store = np.zeros(nt) # the final predictions
    test_set_predictions_credibles = np.zeros((nt, 2)) # the 95% credible interval around predictions    
        
    # count and sample are loop counters within the MCMC
    count=0; sample=0
    
    # prop and acc will store proposal and acceptance rates of the various MCMC moves
    acc=np.zeros(4); prop=np.zeros(4)    
    sample_thin = 0; a_seq = np.zeros((nt, int(cred_n)))
    
    # now for the main program body....
    
    # while we don't have enough samples......keep looping
    while sample < mcmc_samples:
       
        # increment a counter
        count = count+1
        
        # display statistics every 100 iterations
        #print(count)
        if count%1000 == 0:
            print('Its %d Collected %d/%d Acc %.3f L %.3f k %d Prec %f \n' % (count, \
                sample, mcmc_samples, np.sum(acc)/np.sum(prop), marg_lik, k, prec))
        
        # at each iteration: first make a copy of the current model
        beta_prop = np.array(beta)
        X_prop = np.array(X_mars[:, :k])
        Xt_prop = np.array(Xt_mars[:, :k])
        k_prop = int(k)
        if SAVE_SAMPLES:
            basis_params_prop = list(basis_parameters)
        
        #.....anything with a _prop extension is used to denote it as a proposal
        
        # now choose a move
        birth=0; death=0; move=0   # no move chosen yet
        u = np.random.uniform(0, 1, 1) # uniform random variable on U(0,1)
        if u < 0.33:
            # add a basis function
            birth=1; flag=1
            # check for boundary, not allowed more than k_max
            if k == k_max:
                birth=0; move=1; flag=3 # make a "move" move instead
        else:
            if u < 0.66:
                # delete a basis function
                death=1; flag=2
                # check for boundary, not allowed to delete the intercept
                if k == 1:
                    death=0; move=0; flag=3 # note move is set to zero! as we will just re-draw beta if k==1
            else:
                # move a basis function
                move=1; flag=3
                if k == 1:
                    move=0 # just re-draw coefficient

        # store which move we are attempting
        prop[flag-1] = prop[flag-1] + 1
        
        # now depending on move type update the model
        if birth:
            # we're adding a basis function
            k_prop = k + 1
            # choose a random depth for the new basis function
            temp = np.random.uniform(0, 1, 1)
            indx_d = np.ceil(temp*n_inter)
            # choose a random order
            temp = np.random.uniform(0, 1, 1)
            indx_o = np.ceil(temp*n_order)
            # update design matrix with a draw from a Mars basis function
            X_prop_temp, Xt_prop_temp, basis_prop = gen_mars_basis(X, Xt, interaction[int(indx_d-1)], order[int(indx_o-1)])
            X_prop = np.hstack((X_prop, X_prop_temp.reshape((n, 1))))
            Xt_prop = np.hstack((Xt_prop, Xt_prop_temp.reshape((nt, 1))))
            if SAVE_SAMPLES:
                # update basis_parameters
                basis_params_prop.append(basis_prop)
           
        else:
            if death:
                # we've lost a basis function
                k_prop = k - 1
                # choose a basis from the model to delete, NOT THE INTERCEPT THOUGH
                temp = np.random.uniform(0, 1, 1)
                indx = np.ceil(temp*(k-1)) + 1
                # update design matrix
                X_prop = np.delete(X_prop, int(indx-1), 1)
                Xt_prop = np.delete(Xt_prop, int(indx-1), 1)
                if SAVE_SAMPLES:
                    del basis_params_prop[int(indx-1)] 
            if move: 
                # choose a basis from the model to swap with another in dictionary, not the intercept
                temp = np.random.uniform(0, 1, 1)
                indx = np.ceil(temp*(k-1))+1;
                # choose a depth for the new basis function
                temp = np.random.uniform(0, 1, 1)
                indx_d = np.ceil(temp*n_inter);
                # choose an order for the new basis function
                temp = np.random.uniform(0, 1, 1)
                indx_o = np.ceil(temp*n_order);
                # update design matrix
                X_prop[:, int(indx-1)], Xt_prop[:, int(indx-1)], basis_prop = gen_mars_basis(X, Xt, \
                                                    interaction[int(indx_d-1)], order[int(indx_o-1)])
                if SAVE_SAMPLES:
                    # update basis function parameters
                    basis_params_prop[int(indx-1)] = basis_prop

        # get marginal log likelihood of proposed model and a draw of coefficients
        marg_lik_prop, beta_prop, beta_mean_prop, alpha_star_prop = get_ml(X_prop[:, :k_prop], Y, YtY, sig2, alpha_1, alpha_2, prec)
        
        # now see if we accept the proposed change to the model using ratio of probabilities.
        # note that as we draw a new basis function from the prior we only need marginal likelihoods
        rand = np.random.uniform(0, 1, 1)
        if rand < np.exp(marg_lik_prop - marg_lik):
             # we accept the proposed changes: hence update the state of the Markov chain
             beta = np.array(beta_prop)
             beta_mean = np.array(beta_mean_prop)
             alpha_star = float(alpha_star_prop)
             if SAVE_SAMPLES:
                 basis_parameters = list(basis_params_prop)
             
             k = int(k_prop)
             X_mars[:, :k] = np.array(X_prop)
             Xt_mars[:, :k] = np.array(Xt_prop)
             acc[flag-1] = acc[flag-1] + 1
             marg_lik = float(marg_lik_prop)
        
        # update prior precision on beta every 10 iterations after first 200 mcmc its
        if count%10 == 0 and count > 200 and k > 1:
           # get sum squared value of coefficients
           sumsq = np.sum(beta[1:k, ]**2)
           # prec = (1/(0.05+0.5*(1/sig2)*sumsq))*randgamma_mat(0.05+0.5*(k-1),1,1)
           prec = np.random.gamma(shape = 0.05+0.5*(k-1), scale = (1/(0.05+0.5*(1/sig2)*sumsq)), size = 1)
           # prior precision has changed and hence marginal likelihood of current model has changed, so recalculate
           marg_lik, beta, beta_mean, alpha_star = get_ml(X_mars[:, :k], Y, YtY, sig2, alpha_1, alpha_2, prec)
        
        # draw a value for the noise variance - this is needed to draw beta in function get_ml()
        # inverse variance is Gamma
        #sig2_inv = (1/(0.5*(alpha_star + alpha_1)))*randgamma_mat(0.5*(n+alpha_2),1,1)
        sig2_inv = np.random.gamma(shape = 0.5*(n+alpha_2), scale = (1/(0.5*(alpha_star + alpha_1))), size = 1)
        sig2 = 1/sig2_inv
        
        if count > burn_in:
            # start collecting samples
            sample = sample + 1
            
            # get mean predictions
            if sample % thin == 0:
                basis_parameters_seq.append(basis_parameters)
                a = Xt_mars[:, :k].dot(beta_mean[:k, ]) # using the posterior mean of beta
                
                # store statistics
                pred_store = pred_store + a
                chain_stats_k_store[sample_thin] = int(k)
                chain_stats_LL_store[sample_thin] = float(marg_lik)
                
                # store credibles
                a = Xt_mars[:, :k].dot(beta[:k, ]) # using draw of beta not mean of beta
                a_seq[:, sample_thin] = np.array(a)
                sample_thin = sample_thin + 1
                 
    # end the mcmc loop
    
    # get 95% interval
    cred_upper = np.percentile(a_seq, 97.5, axis=1)
    cred_lower = np.percentile(a_seq, 2.5, axis=1)
    
    # get MCMC mean
    #test_set_predictions_pred_store = pred_store/cred_n
    test_set_predictions_pred_store = np.mean(a_seq, axis = 1)
    
    # check the final test error and display
    pred_t = pred_store/sample
    test_er = np.sum((Yt-pred_t)**2)
    print('Final Test er %.3f \n' % test_er)

    # calculate credibles
    #test_set_predictions_credibles = np.array((min_cred_upper, max_cred_lower))
    test_set_predictions_credibles = np.hstack((cred_upper.reshape(nt, 1), cred_lower.reshape(nt, 1)))
    
    return test_set_predictions_credibles, test_set_predictions_pred_store, chain_stats_k_store, \
           chain_stats_LL_store, a_seq, basis_parameters_seq


#################################### other functions needed by the main function
            
# function that gets marginal likelihood and draws beta 
def get_ml(X, Y, YtY, sig2, a, b, prec):
    """
    function to calculate marginal likelihood of Bayes linear model, Y ~ N(X beta, sig2 I)
    with normal-inverse-gamma prior on beta, sig2 ~ NIG(0,prec I, a, b)
      
    Parameters
    ----------
    X : array
        the design matrix.
    Y : array
        the response.
    YtY : float
        the sum squared of response values, Y.T.dot(Y).
    sig2 : float
        a draw from the noise variance.
    a : float
        prior parameters for noise variance.
    b : float
        prior parameters for noise variance.
    prec : float
        precision of normal prior on beta: beta | sig2 ~ N(0, sig2 * (1/prec) * I).

    Returns
    -------
	log_ML - log marginal likelihood (up to a constant)
	beta - a draw from the posterior distribution of beta
	beta_mean - the posterior mean vector for beta
	a_star - the posterior sum_squares
    """
    
    import numpy as np
    import random
    import scipy.linalg
    from scipy.stats import invgauss
    import os
    import pickle
    
    n, p = np.shape(X)
    
    # make prior precision (inverse-variance) matrix......
    prior_prec = prec*np.identity(p)
    prior_prec[0, 0] = 0 # improper prior on intercept (first col of X)
    
    # calculate posterior variance covariance matrix and precision
    post_P = X.T.dot(X) + prior_prec
    post_V = np.linalg.pinv(post_P)
    
    # get posterior mean of beta
    beta_mean = post_V.dot(X.T.dot(Y))
    
    # calculate log of the square root of determinant of post_V by using Cholesky decomposition
    R = np.linalg.cholesky(post_V).T
    # this is nice as the log of square root of determinant of post_V is just the 
    # sum of the log of the diagonal elements of R, where post_V = R'*R, R is upper triangular
    half_log_det_post = np.sum(np.log(np.diag(R)))
    
    # now calculate log of square root of determinant of prior (this is easy as prior on beta is diagonal) 
    half_log_det_prior = -0.5*(p-1)*np.log(prec)
    #.......note that we use (p-1) as we use improper prior on intecept, beta(1) ~ N(0, infinity)
    #.....this does not cause (Lyndley-Bartlett paradox) problems as we allways include an intercept in the model
    
    # now calculate posterior sum_squares
    a_star = YtY - beta_mean.T.dot(post_P).dot(beta_mean)
    
    # finally log marginal likelihood is
    log_ML = half_log_det_post - half_log_det_prior - (0.5*(n+b))*np.log(0.5*(a+a_star))
    #log_ML = half_log_det_post - half_log_det_prior - 0.5*(a+a_star) - (0.5*(n+b))*np.log(sig2)
    
    # Now draw a value of beta from conditional posterior distribution....
    # making use of previous cholesky decomposition
    Rsig2 = np.sqrt(sig2)*R
    Rsig2 = Rsig2.T
    randn = np.random.normal(0, 1, p)
    beta =  beta_mean + Rsig2.dot(randn)  
    
    return log_ML, beta, beta_mean, a_star

# function that generates a MARS basis function   

def gen_mars_basis(X, Xt, interaction, order):  
    """
    generates random mars basis

    Parameters
    ----------
    X : array
        data.
    Xt : array
        OPTIONAL data for prediction of basis on.
    interaction : integer
        number of terms-1 in the basis function.
    order : TYPE
         order of basis (typically order \in {0 1}, i.e. step or linear).

    Returns
    -------
    x - basis response on X
    xt - basis response on Xt
    basis - stores basis parameters 

    """

    import numpy as np
    import random
    import scipy.linalg
    from scipy.stats import invgauss
    import os
    import pickle

    # get data sizes
    n, p = X.shape
    
    # depth stores intreraction level. Need to add one to this to get 
    # number of terms in the basis function
    depth = interaction + 1
    
    # check for integrety
    if depth > p:
        depth = p
    
    # we need to calculate response on test data set
    test = 1
    if test:
        nt = Xt.shape[0]
    
    # knot_pos stores knot position
    knot = np.zeros(depth)
    
    # lr stores whether each knot is `left' or "right" facing
    lr = -1 * np.ones(depth)
    
    # reponse of basis function stored in 
    x = np.zeros((n, 1))
    if test:
        xt = np.zeros((nt, 1))
    
    # now make basis function
    
    # make temp response
    temp = np.zeros((n, depth))
    if test:
        temp_t = np.zeros((nt, depth))
    
    # repeat until we get a non-zero basis function
    not_finished = 1
    numite = 1
    while not_finished:
        #print(numite)
        numite = numite + 1
       
        # var stores indicator of which covariates are used
        var = np.zeros((depth, 1))
       
        for j in range(depth):
            #print(j)
            # choose a data point to locate knot on
            rand = np.random.uniform(0, 1, 1)
            data_indx = np.ceil(rand*n);
            
            # choose a variable not already used
            not_ok = 1
            while not_ok:
                rand = np.random.uniform(0, 1, 1)
                ind = np.ceil(rand*p)
                if ind not in var[:(j+1)]:
                    var[j] = ind
                    not_ok = 0

            
            # choose left/right for the knot
            rand = np.random.uniform(0, 1, 1)
            lr[j] = rand > 0.5
            
            # choose knot position
            knot[j] = X[int(data_indx-1), int(var[j]-1)]
            
            temp_xj = np.zeros((n, 2))
            temp_xj[:, 1] = X[:, int(var[j]-1)]-knot[j]
            if test:
                temp_xtj = np.zeros((nt, 2))
                temp_xtj[:, 1] = Xt[:, int(var[j]-1)]-knot[j]                
            if lr[j] == 0:
                temp[:, j] = - np.min(temp_xj, axis=1)
                if test:
                   temp_t[:, j] = - np.min(temp_xtj, axis=1)
            else:
                temp[:, j] = np.max(temp_xj, axis=1)
                if test:
                    temp_t[:, j] = np.max(temp_xtj, axis=1)

        # put to power
        if order == 0:
            temp = temp != 0
            if test:
                temp_t = temp_t != 0
        else:
            temp = temp**order
            if test:
                temp_t = temp_t**order

        # tensor product
        x = np.prod(temp, axis=1)
        if test:
            xt = np.prod(temp_t, axis=1)
        else:
            xt = 0
        
        # null basis functions
        stx = np.std(x, axis=0)
        not_finished = (np.mean(x == 0) == 1) or (np.mean(stx == 0) != 0)
        
    
    # standardise function
    mx = np.mean(x, axis=0)
    #stx = np.std(x, axis=0)
    x = (x-mx)/stx
    xt = (xt-mx)/stx
    
    # define a class to store the basis function parameters
    class basis_para:
        def __init__(self, order, inter, knot, var, lr, mx, sx):
            self.order = order
            self.inter = inter
            self.knot = knot
            self.var = var 
            self.lr = lr
            self.mx = mx
            self.sx = sx  
    
    # store basis function paramters
    basis = basis_para(order, depth, knot, var, lr, mx, stx)

    return x, xt, basis      
    
    
    
    
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = np.linspace(0,10,10)
    y = np.sin(x)
    
    plt.scatter(x,y)
    
    gp = gp_model(x, y, [1], 1, 0.05, 1, "SE")
    
    x_plot = np.linspace(0,10,1000)
    
    mean, var = gp.predict_var(x_plot)
    
    plt.fill_between(x_plot, mean+np.sqrt(var), mean-np.sqrt(var), alpha=0.5)
    plt.plot(x_plot, mean)
    
    