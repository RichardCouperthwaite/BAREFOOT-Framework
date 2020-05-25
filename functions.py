# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:25:53 2019

@author: Richard Couperthwaite

This code is to achieve two objectives:
    1) Provide the code for various microstructure/grain size models or interfaces
    to the code that will run such models
    2) Provide the code for using the Reification method in Python. Will use the
    george.py module for the GP fit.
"""
from george import kernels, GP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        if self.kern == 'SE':
            return self.sigma_f * kernels.ExpSquaredKernel(self.l_param, ndim=self.n_dim)
        elif self.kern == 'M32':
            return self.sigma_f * kernels.Matern32Kernel(self.l_param, ndim=self.n_dim)
        elif self.kern == 'M52':
            return self.sigma_f * kernels.Matern52Kernel(self.l_param, ndim=self.n_dim)
    
    def create_gp(self):
        gp = GP(kernel=self.kk, mean=self.mean)
        gp.compute(self.x_train, self.sigma_n)
        return gp
    
    def predict_cov(self, x_pred):
        mean, sigma = self.gp.predict(self.y_train, x_pred, kernel = self.kk, return_cov=True, return_var=False)
        return mean, sigma
    
    def predict_var(self, x_pred):
        mean, var = self.gp.predict(self.y_train, x_pred, kernel = self.kk, return_cov=False, return_var=True)
        return mean, var
    
    def update(self, new_x_data, new_y_data, new_y_err, err_per_point):
        self.x_train = np.vstack((self.x_train, new_x_data))
        self.y_train = np.append(self.y_train, new_y_data)
        # self.x_train = np.append(self.x_train, new_x_data)
        # self.y_train = np.append(self.y_train, new_y_data)
        if err_per_point:
            self.sigma_n = np.append(self.sigma_n, new_y_err)
            
        self.gp = self.create_gp()
        
    def log_likelihood(self):
#        return self.gp.lnlikelihood(self.y_train, quiet=True)
        return self.gp.log_likelihood(self.y_train, quiet=True)
    
    def get_hyper_params(self):
        curr_params = self.gp.get_parameter_vector()
        params = []
        for i in range(len(curr_params)):
            if i == 0:
                params.append(np.exp(curr_params[i])*self.n_dim)
            else:
                params.append(np.sqrt(np.exp(curr_params[i])))
        return np.array(params)
        
    def hp_optimize(self, meth="L-BFGS-B", update=False):
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
        
        
def reification(y, sig):
    """
    This function is coded to enable the reification of any number of models.
    """
    mean_fused = []
    var_fused = []
    eps = 0.1
    
    rtest = []
        
    rho_bar = []
    for i in range(len(y)-1):
        for j in range(len(y)-i-1):
#            y[j+i][np.where((np.abs(y[i]-0)<eps)*(np.abs(y[j+i]-0)<eps))] = 10
            
            rho1 = np.divide(np.sqrt(sig[i]), np.sqrt((y[i]-y[j+i+1])**2 + sig[i]))
            rtest.append(rho1)
            rho2 = np.divide(np.sqrt(sig[j+i+1]), np.sqrt((y[j+i+1]-y[i])**2 + sig[j+i+1]))
            rtest.append(rho2)
            rho_bar_ij = np.divide(sig[j+i+1], (sig[i]+sig[j+i+1]))*rho1 + np.divide(sig[i], (sig[i]+sig[j+i+1]))*rho2
            rho_bar_ij[np.where(rho_bar_ij>0.99)] = 0.99
            rho_bar.append(rho_bar_ij)
            
    mm = rho_bar[0].shape[0]
            
    sigma = np.zeros((len(y), len(y)))
    
    for i in range(mm):
        for j in range(len(y)):
            for k in range(len(y)-j):
                jj = j
                kk = k+j
                if jj == kk:
                    sigma[jj,kk] = sig[jj][i]
                else:
                    sigma[jj,kk] = rho_bar[kk+jj-1][i]*np.sqrt(sig[jj][i]*sig[kk][i])
                    sigma[kk,jj] = rho_bar[kk+jj-1][i]*np.sqrt(sig[jj][i]*sig[kk][i])

        alpha = np.linalg.inv(sigma)
        w = np.sum(alpha,1)/np.sum(alpha)
        mu = y[0][i]
        for j in range(len(y)-1):
            mu = np.hstack((mu,y[j+1][i]))
        mean = np.sum(w@mu)
        
        mean_fused.append(mean)
        var_fused.append(1/np.sum(alpha))
        
    return np.array(mean_fused), np.array(var_fused)

def knowledge_gradient(M, sn, mu, sigma):
    """
    This is the method used to determine the knowledge gradient of the fused model
    for a given set of test data points. The aim is to calculate the best possible
    point to be used in the next iteration. This function will be called after the
    fused model is calculated for each of the lower order models being assumed to
    be the truth model.
    
    Implementation based on the work by Frazier, Powell, Dayanik
    [1]P. Frazier, W. Powell, and S. Dayanik, “The Knowledge-Gradient Policy for Correlated Normal Beliefs,” INFORMS Journal on Computing, vol. 21, no. 4, pp. 599–613, May 2009.
    
    M: the number of samples
    sn: the noise of the model
    mu: mean of the model for all M samples
    sigma: covariance matrix of the model
    
    The function returns:
    NU: Knowledge Gradient values for all the samples
    nu_star: the maximum knowledge gradient value
    x_star: the index of the value with the maximum knowledge gradient (0 as first index)
    """
    from scipy.stats import norm
    
    def algorithm1(a, b, M):
        c = [np.inf]
        A = [0]
        for i in range(M-1):
            c.append(np.inf)
            t = 0
            while t == 0:
                j = A[-1]
                c[j] = (a[j]-a[i+1])/(b[i+1]-b[j])
                if (len(A)!=1) and (c[j]<=c[A[-2]]):
                    A = A[0:-1] 
                else:
                    t = 1
            A.append(i+1)
        c = np.array(c)
        A = np.array(A)
        return c, A
    
    NU = []
    
    for i in range(M):   
        a = mu
        try:
            b = sigma[:,i]/np.sqrt(sn**2+sigma[i,i])
        except IndexError:
            b = sigma/np.sqrt(sn**2+sigma[i])
            
        I = np.argsort(b)
        a = a[I]
        b = b[I]
        bb, indexes, inverse = np.unique(b, return_index=True, return_inverse=True)
        aa = []
        for ii in range(len(indexes)):
            aa.append(np.max(a[np.where(b == b[indexes[ii]])]))
            
        MM = len(aa)
        aa = np.array(aa)
        c, A = algorithm1(aa, bb, MM)
        aa = aa[A]
        bb = bb[A]
        c = c[A]
        MM = A.shape[0]
        sig = 0
        for ii in range(MM-1):
            sig += (bb[ii+1]-bb[ii])*(norm.pdf(-abs(c[ii]))+ (-abs(c[ii])) * norm.cdf(-abs(c[ii])))
        nu = np.log(sig)
        NU.append(nu)
        
        try:
            if nu>nu_star:
                nu_star = nu
                x_star = i
        except NameError:
            nu_star = nu
            x_star = i
    
    return nu_star, x_star, NU

def KG_cost_optimization(models, err_models, current_model_index, x_alt, x_test, prior_error, prior, sn, costs):
    model_mean, model_var = models[current_model_index].predict_var(x_alt)
    model_std = model_var**(0.5)
#    print(model_var)
    
    NU_avg = []
    MAX_avg = []
    for aa in range(x_alt.shape[0]):
        nu = []
        maxval = []
        normsamples = np.random.normal(loc=model_mean[aa], scale=model_std[aa], size=15)
        for bb in range(15):
            GP_temp = deepcopy(models[current_model_index])
            GP_temp.update(x_alt[aa], normsamples[bb], sn[current_model_index+1], False)            
            y_new = []
            v_new = []
            
            for i in range(len(models)):
                if i == current_model_index:
                    y_pred, y_var = GP_temp.predict_var(x_test)
                else:
                    y_pred, y_var = models[i].predict_var(x_test)
                y_err_pred, y_err_var = err_models[i].predict_var(x_test)
                y_err_pred += prior_error
                y_new.append(y_pred)
                v_new.append((y_err_pred)**2 + y_var)
            
            mean_fused, var_fused = reification(y_new, v_new)
            
            nu_star, x_star, NU = knowledge_gradient(x_test.shape[0], sn[current_model_index+1], mean_fused, np.diag(np.abs(var_fused)))
            
            nu.append(np.exp(nu_star))
            maxval.append(np.max(mean_fused))
        
        NU_avg.append(np.mean(nu))
        MAX_avg.append(np.mean(maxval))
    
    NU_avg = np.array(NU_avg)
    MAX_avg = np.array(MAX_avg)
    KG_corrected = ((NU_avg+MAX_avg)/costs[current_model_index])
    
    KG_out = np.max(KG_corrected)
    x_out = np.where(KG_corrected == KG_out)[0]
    
    return KG_out, x_out, NU_avg, MAX_avg, KG_corrected, NU, nu_star, x_star


class model_reification():
    def __init__(self, x_train, y_train, l_param, sigma_f, sigma_n, l_param_err, 
                 sigma_f_err, sigma_n_err, x_true, y_true, num_models, num_dim, 
                 kernel, prior, prior_error):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.l_param = np.array(l_param)
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.l_param_err = np.array(l_param_err)
        self.sigma_f_err = np.array(sigma_f_err)
        self.sigma_n_err = np.array(sigma_n_err)
        self.x_true = np.array(x_true)
        self.y_true = np.array(y_true)
        self.num_models = num_models
        self.num_dim = num_dim
        self.kernel = kernel
        self.prior = prior
        self.prior_error = prior_error
        self.gp_models = self.create_gps()
        self.gp_err_models = self.create_error_models()
        self.fused_GP = ''
        self.fused_l = ''
        self.fused_sn = ''
        self.fused_sf = ''
        self.fused_kernel = ''
        
    def create_gps(self):
        """
        GPs need to be created for each of the lower dimension information sources
        as used in the reification method. These can be multi-dimensional models.
        As a result, the x_train and y_train data needs to be added to the class
        as a list of numpy arrays.
        """
        gp_models = []
        for i in range(self.num_models):
            new_model = gp_model(self.x_train[i], self.y_train[i]-self.prior, self.l_param[i], 
                                 self.sigma_f[i], self.sigma_n[i], self.num_dim, self.kernel)
            gp_models.append(new_model)
        return gp_models
    
    def create_error_models(self):
        """
        In order to calculate the total error of the individual GPs an error
        model is created for each GP. The inputs to this are the error between
        the individual GP predictions and the truth value at all available truth
        data points. The prior_error value is subtracted from the difference to
        ensure that the points are centred around 0.
        """
        gp_error_models = []
        for i in range(self.num_models):
            gpmodel_mean, gpmodel_var = self.gp_models[i].predict_var(self.x_true)
            error = np.abs(self.y_true-gpmodel_mean) - self.prior_error
            new_model = gp_model(self.x_true, error, self.l_param_err[i], self.sigma_f_err[i], 
                                 self.sigma_n_err[i], self.num_dim, self.kernel)
            gp_error_models.append(new_model)
        return gp_error_models
    
    def create_fused_GP(self, x_test, l_param, sigma_f, sigma_n, kernel):
        model_mean = []
        model_var = []
        for i in range(len(self.gp_models)):
            m_mean, m_var = self.gp_models[i].predict_var(x_test)
            model_mean.append(m_mean+self.prior)
            err_mean, err_var = self.gp_err_models[i].predict_var(x_test)
            model_var.append((err_mean+self.prior_error)**2 + m_var)
        fused_mean, fused_var = reification(model_mean, model_var)
        self.fused_GP = gp_model(x_test[0:200:12], fused_mean[0:200:12], l_param, sigma_f, abs(fused_var[0:200:12])**(0.5), self.num_dim, kernel)
        
    def update_GP(self, new_x, new_y, model_index):
        np.column_stack((self.x_train[model_index], new_x))
        np.column_stack((self.y_train[model_index], new_y-self.prior))
        self.gp_models[model_index].update(new_x, new_y, self.sigma_n[model_index], False)
    
    def update_truth(self, new_x, new_y):
        np.column_stack((self.x_true, new_x))
        np.column_stack((self.y_true, new_y-self.prior))
        self.gp_err_models = self.create_error_models()
    
    
        
            


if __name__ == "__main__":
    data = pd.read_csv('DEMS_Paper1_Data.csv', )
#    print(data.head())
#    
#    plt.fill_between(data['Volume Fraction Martensite'], data['RVE']+data['RVE Error'], data['RVE']-data['RVE Error'], color="k", alpha=0.2)
#    plt.plot(data['Volume Fraction Martensite'], data['RVE'], color="k", label="RVE")
#    plt.plot(data['Volume Fraction Martensite'], data['Isostrain'], color="b", label="Isostrain")
#    plt.plot(data['Volume Fraction Martensite'], data['Isostress'], color="r", label="Isostress")
#    plt.plot(data['Volume Fraction Martensite'], data['Isowork'], color="g", label="Isowork")
#    plt.xlabel('Volume Fraction Martensite')
#    plt.ylabel(r'Normalized Strain Hardening Rate ($\frac{1}{\sigma}\frac{d\sigma}{d\epsilon}$)')
#    plt.legend()
#    plt.show()
    
    sample = [0,12,24,36,48,60,72,84,96]
#    sample = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    test_points = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100]
    test_data = data.iloc[test_points]
    x_test = test_data.iloc[:,[0]]
    uniform_data = data.iloc[sample]
    uniform_data
    
    l1 = 40
    l2 = 40
    l3 = 40
    l4 = 40
    sf1 = 0.5
    sf2 = 0.5
    sf3 = 0.5
    sf4 = 0.5
    sn1 = 0
    sn2 = 0
    sn3 = 0
    sn4 = 0
    
    # Parameters for the GPs are:
    # x_train, y_train, l_param, sigma_f, sigma_n, n_dim, kern
    
    names = []
    MSE = []
    
    isostrain_GP = gp_model(uniform_data['Volume Fraction Martensite'], uniform_data['Isostrain'], [l1], sf1, sn1, 1, 'SE')
    isostress_GP = gp_model(uniform_data['Volume Fraction Martensite'], uniform_data['Isostress'], [l2], sf2, sn2, 1, 'SE')
    isowork_GP = gp_model(uniform_data['Volume Fraction Martensite'], uniform_data['Isowork'], [l3], sf3, sn3, 1, 'SE')
    RVE_GP = gp_model(uniform_data['Volume Fraction Martensite'], uniform_data['RVE'], [l4], sf4, sn4, 1, 'SE')

    pred, pred_var = isostrain_GP.predict_var(test_data['Volume Fraction Martensite'])
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), pred-2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), color="k", alpha=0.2)
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var), pred-2*np.sqrt(pred_var), color="k", alpha=0.5)
    plt.plot(test_data['Volume Fraction Martensite'], pred, color="k", label="Isostrain")
    plt.plot(data['Volume Fraction Martensite'], data['RVE'], color='g')
    plt.scatter(uniform_data['Volume Fraction Martensite'], uniform_data['Isostrain'])
    plt.ylim([-5,55])
    plt.show()
    
    names.append('Isostrain')
    MSE.append((1/test_data['Volume Fraction Martensite'].shape[0])*(np.sum(test_data['RVE']-pred)**2))
    
    pred, pred_var = isostress_GP.predict_var(test_data['Volume Fraction Martensite'])
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), pred-2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), color="k", alpha=0.2)
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var), pred-2*np.sqrt(pred_var), color="k", alpha=0.5)
    plt.plot(test_data['Volume Fraction Martensite'], pred, color="k", label="Isostress")
    plt.plot(data['Volume Fraction Martensite'], data['RVE'], color='g')
    plt.scatter(uniform_data['Volume Fraction Martensite'], uniform_data['Isostress'])
    plt.ylim([-5,55])
    plt.show()
    
    names.append('Isostress')
    MSE.append((1/test_data['Volume Fraction Martensite'].shape[0])*(np.sum(test_data['RVE']-pred)**2))
    
    pred, pred_var = isowork_GP.predict_var(test_data['Volume Fraction Martensite'])
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), pred-2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), color="k", alpha=0.2)
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var), pred-2*np.sqrt(pred_var), color="k", alpha=0.5)
    plt.plot(test_data['Volume Fraction Martensite'], pred, color="k", label="Isowork")
    plt.plot(data['Volume Fraction Martensite'], data['RVE'], color='g')
    plt.scatter(uniform_data['Volume Fraction Martensite'], uniform_data['Isowork'])
    plt.ylim([-5,55])
    plt.show()
    
    names.append('Isowork')
    MSE.append((1/test_data['Volume Fraction Martensite'].shape[0])*(np.sum(test_data['RVE']-pred)**2))
    
    pred, pred_var = RVE_GP.predict_var(test_data['Volume Fraction Martensite'])
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), pred-2*np.sqrt(pred_var+np.abs(test_data['RVE']-pred)), color="k", alpha=0.2)
    plt.fill_between(test_data['Volume Fraction Martensite'], pred+2*np.sqrt(pred_var), pred-2*np.sqrt(pred_var), color="k", alpha=0.5)
    plt.plot(test_data['Volume Fraction Martensite'], pred, color="k", label="RVE")
    plt.plot(data['Volume Fraction Martensite'], data['RVE'], color='g')
    plt.scatter(uniform_data['Volume Fraction Martensite'], uniform_data['RVE'])
    plt.ylim([-5,55])
    plt.show()
    
    names.append('RVE_GP')
    MSE.append((1/test_data['Volume Fraction Martensite'].shape[0])*(np.sum(test_data['RVE']-pred)**2))

    
    
    fused_mean, fused_var = reification([isostress_GP, isostrain_GP, isowork_GP], test_data['Volume Fraction Martensite'], test_data['RVE'])
    plt.fill_between(test_data['Volume Fraction Martensite'], fused_mean+2*np.sqrt(fused_var), fused_mean-2*np.sqrt(fused_var), color="k", alpha=0.2)
    plt.plot(test_data['Volume Fraction Martensite'], fused_mean, color="k", label="Isostrain")
    plt.plot(data['Volume Fraction Martensite'], data['RVE'], color='g')
    plt.ylim([-5,55])
#    plt.scatter(data['Volume Fraction Martensite'], data['RVE'])
    plt.show()
    
    names.append('Fused')
    MSE.append((1/test_data['Volume Fraction Martensite'].shape[0])*(np.sum(test_data['RVE']-fused_mean)**2))
    
    print(names)
    print(MSE)
    
    
    
    
#    x_test = np.linspace(0,100,num=100)
#    fused_mean, fused_var = reification([RVE_GP, isostrain_GP, isowork_GP, isostress_GP], x_test)
#    plt.fill_between(x_test, fused_mean+2*np.sqrt(fused_var), fused_mean-2*np.sqrt(fused_var), color="k", alpha=0.2)
#    plt.plot(x_test, fused_mean, color="k", label="Isostrain")
#    plt.scatter(data['Volume Fraction Martensite'], data['RVE'])
#    plt.show()