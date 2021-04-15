# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:48:44 2020

@author: richardcouperthwaite
"""

import numpy as np
from gpModel import gp_model
import matplotlib.pyplot as plt

def reification(y,sig):    
    # This function takes lists of means and variances from multiple models and 
    # calculates the fused mean and variance following the Reification/Fusion approach
    # developed by D. Allaire. This function can handle any number of models.
    y = np.array(y)
    yM = np.transpose(np.tile(y, (len(y),1,1)), (2,0,1))
    sigM = np.transpose(np.tile(sig, (len(y),1,1)), (2,0,1))
    
    unoM = np.tile(np.diag(np.ones(len(y))), (yM.shape[0],1,1))
    zeroM = np.abs(unoM-1)
    
    yMT = np.transpose(yM, (0,2,1))
    sigMT = np.transpose(sigM, (0,2,1))
    
    # The following individual value calculations are compacted into the single calculation
    # for alpha, but are left here to aid in understanding of the equation
    
    rho1 = np.divide(np.sqrt(sigM), np.sqrt((yM-yMT)**2 + sigM))
    rho2 = np.sqrt(sigMT)/np.sqrt((yMT-yM)**2 + sigMT)
    rho = (sigMT/(sigMT+sigM))*rho1 + (sigM/(sigMT+sigM))*rho2
    sigma = rho*zeroM*(np.sqrt(sigM*sigMT)) + unoM*sigM
    alpha = np.linalg.pinv(sigma)
    
    # alpha = np.linalg.pinv(((sigM/(sigM+sigMT))*np.sqrt(sigM)/np.sqrt((yM-yMT)**2 + sigM) + \
    #     (sigMT/(sigM+sigMT))*np.sqrt(sigMT)/np.sqrt((yMT-yM)**2 + sigMT))*zeroM*(np.sqrt(sigM*sigMT)) + unoM*sigM)
    
    w = (np.sum(alpha,2))/np.tile(np.sum(alpha,axis=(1,2)), (len(y),1)).transpose()
    
    mean= np.sum(w*(y.transpose()), axis=1)
    var = 1/np.sum(alpha,axis=(2,1))
    
    return mean, var

class model_reification():
    """
    This python class has been developed to aid with the reification/fusion approach.
    The class provides methods that automate much of the process of defining the 
    discrepancy GPs and calculating the fused mean and variance
    """
    def __init__(self, x_train, y_train, l_param, sigma_f, sigma_n, model_mean,
                 model_std, l_param_err, sigma_f_err, sigma_n_err, 
                 x_true, y_true, num_models, num_dim, kernel):
        self.x_train = x_train
        self.y_train = y_train
        self.model_mean = model_mean
        self.model_std = model_std
        self.model_hp = {"l": np.array(l_param),
                         "sf": np.array(sigma_f),
                         "sn": np.array(sigma_n)}
        self.err_mean = []
        self.err_std = []
        self.err_model_hp = {"l": np.array(l_param_err),
                             "sf": np.array(sigma_f_err),
                             "sn": np.array(sigma_n_err)}
        self.x_true = x_true
        self.y_true = y_true
        self.num_models = num_models
        self.num_dim = num_dim
        self.kernel = kernel
        self.gp_models = self.create_gps()
        self.gp_err_models = self.create_error_models()
        self.fused_GP = ''
        self.fused_y_mean = ''
        self.fused_y_std = ''
        
    def create_gps(self):
        """
        GPs need to be created for each of the lower dimension information sources
        as used in the reification method. These can be multi-dimensional models.
        As a result, the x_train and y_train data needs to be added to the class
        as a list of numpy arrays.
        """
        gp_models = []
        for i in range(self.num_models):
            new_model = gp_model(self.x_train[i], 
                                 (self.y_train[i]-self.model_mean[i])/self.model_std[i], 
                                 self.model_hp["l"][i], 
                                 self.model_hp["sf"][i], 
                                 self.model_hp["sn"][i], 
                                 self.num_dim, 
                                 self.kernel)
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
            gpmodel_mean = gpmodel_mean * self.model_std[i] + self.model_mean[i]
            error = np.abs(self.y_true-gpmodel_mean)
            self.err_mean.append(np.mean(error))
            self.err_std.append(np.std(error))
            if self.err_std[i] == 0:
                self.err_std[i] = 1
            new_model = gp_model(self.x_true, 
                                 (error-self.err_mean[i])/self.err_std[i], 
                                 self.err_model_hp["l"][i], 
                                 self.err_model_hp["sf"][i], 
                                 self.err_model_hp["sn"][i], 
                                 self.num_dim, 
                                 self.kernel)
            gp_error_models.append(new_model)
        return gp_error_models
    
    def create_fused_GP(self, x_test, l_param, sigma_f, sigma_n, kernel):
        """
        In this function we create the fused model by calculating the fused mean
        and variance at the x_test values and then fitting a GP model using the
        given hyperparameters
        """
        model_mean = []
        model_var = []
        for i in range(len(self.gp_models)):
            m_mean, m_var = self.gp_models[i].predict_var(x_test)
            m_mean = m_mean * self.model_std[i] + self.model_mean[i]
            m_var = m_var * (self.model_std[i] ** 2)
            model_mean.append(m_mean)
            err_mean, err_var = self.gp_err_models[i].predict_var(x_test)
            err_mean = err_mean * self.err_std[i] + self.err_mean[i]
            model_var.append((err_mean)**2 + m_var)
        fused_mean, fused_var = reification(model_mean, model_var)
        self.fused_y_mean = np.mean(fused_mean[0:400:12])
        self.fused_y_std = np.std(fused_mean[0:400:12])
        if self.fused_y_std == 0:
            self.fused_y_std = 1
        fused_mean = (fused_mean - self.fused_y_mean)/self.fused_y_std
        self.fused_GP = gp_model(x_test[0:400:12], 
                                 fused_mean[0:400:12], 
                                 l_param, 
                                 sigma_f, 
                                 abs(fused_var[0:400:12])**(0.5), 
                                 self.num_dim, 
                                 kernel)
        return self.fused_GP
        
    def update_GP(self, new_x, new_y, model_index):
        """
        Updates a given model in the reification object with new training data
        amd retrains the GP model
        """
        self.x_train[model_index] = np.vstack((self.x_train[model_index], new_x))
        self.y_train[model_index] = np.append(self.y_train[model_index], new_y)
        self.gp_models[model_index].update(new_x, new_y, self.model_hp['sn'][model_index], False)
    
    def update_truth(self, new_x, new_y):
        """
        Updates the truth model in the reification object with new training data
        and then recalculates the error models
        """
        self.x_true = np.vstack((self.x_true, new_x))
        self.y_true = np.append(self.y_true, new_y)
        self.gp_err_models = self.create_error_models()
        
    def predict_low_order(self, x_predict, index):
        """
        Provides a prediction from the posterior distribution of one of the 
        low order models
        """
        gpmodel_mean, gpmodel_var = self.gp_models[index].predict_var(x_predict)
        gpmodel_mean = gpmodel_mean * self.model_std[index] + self.model_mean[index]
        gpmodel_var = gpmodel_var * (self.model_std[index]**2)
        return gpmodel_mean, gpmodel_var
    
    def predict_fused_GP(self, x_predict):
        """
        Provides a prediction from the posterior distribution of the Fused Model
        """
        gpmodel_mean, gpmodel_var = self.fused_GP.predict_var(x_predict)
        gpmodel_mean = gpmodel_mean * self.fused_y_std + self.fused_y_mean
        gpmodel_var = gpmodel_var * (self.fused_y_std**2)
        return gpmodel_mean, np.diag(gpmodel_var)
    
            
def reification_old(y, sig):
    """
    This function is coded to enable the reification of any number of models. 
    This function relied on python loops and so was renovated to obtain the function
    above. This is left for potential additional understanding of the computational 
    approach used.
    """
    mean_fused = []
    var_fused = []
    
    rtest = []
        
    rho_bar = []
    for i in range(len(y)-1):
        for j in range(len(y)-i-1):
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
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    y = np.array([[4.300526128, 4.685873164, 4.685320709, 4.684972467, 4.614799791, 4.434835402, 4.181252705, 3.823263849, 3.479196007, 2.993645031, 2.578627079, 2.089518798, 1.69950212, 1.252310983, 0.930222813, 0.548720249, 0.30423359, 0.01530769, -0.188495143, -0.39157984, -0.586544877, -0.750773457, -0.91084303, -1.094303979, -1.294498015, -1.443896999, -1.566636037, -1.745273912, -1.934402268, -2.146824783, -2.269359962, -2.400116708, -2.584035626, -2.693559298, -2.844288792, -2.916172654, -3.026356671, -3.148829538, -3.322578993, -3.479626397, -3.650563892, -3.886141488, -4.053997337, -4.22226307, -4.44463673, -4.695529817, -4.8192348, -4.972121994, -5.069316321, -5.07223674, -5.020473931, -4.887328924, -4.662800379, -4.345402601, -4.00879642, -3.592663527, -3.212924843, -2.724414187, -2.419973138, -1.937232245, -1.626780165, -1.312098818, -1.083850218, -0.806382951, -0.619269895, -0.499698275, -0.387734871, -0.323103065, -0.301200977, -0.316899866, -0.366709982, -0.465173385, -0.617833415, -0.875061492, -1.152133646, -1.539703057, -1.895007352, -2.607029161, -3.03104169, -3.678324538, -4.474932122, -4.975841622, -6.075653604, -6.443567254, -7.19358372, -7.66883659, -8.050657669, -8.312958066, -8.398852375, -8.310414181, -8.06659005, -7.739931222, -7.225175391, -6.631978401, -5.885223218, -5.166929247, -4.426770078, -3.793267722, -3.124223352, -2.552502545, -2.026654555],
                     [4.958517604, 4.181887959, 4.176351388, 4.173025062, 3.892310633, 3.548083213, 3.213989935, 2.838561356, 2.526035971, 2.125697454, 1.80372785, 1.434557882, 1.140724389, 0.796318928, 0.537798707, 0.212415107, -0.01206429, -0.299734791, -0.520859861, -0.758028192, -1.000408328, -1.212226155, -1.420718242, -1.656490953, -1.905133413, -2.083811654, -2.226529814, -2.428974021, -2.638919042, -2.873493989, -3.010283275, -3.159096962, -3.376130711, -3.510677857, -3.701407368, -3.793137571, -3.931291965, -4.07641382, -4.260069254, -4.403769044, -4.541234543, -4.708595049, -4.817943119, -4.922907133, -5.059162247, -5.218236, -5.304282165, -5.431145326, -5.567770579, -5.641373904, -5.710249553, -5.773147022, -5.818123322, -5.839210969, -5.834721658, -5.804404736, -5.757572917, -5.672474213, -5.604854068, -5.471668271, -5.366025128, -5.238706386, -5.130136335, -4.973239079, -4.845444567, -4.749497773, -4.641998051, -4.560647432, -4.505739378, -4.486852441, -4.492378149, -4.530987477, -4.606854189, -4.744262227, -4.89279768, -5.095166536, -5.274155984, -5.616151485, -5.811263742, -6.100113911, -6.445754035, -6.660386772, -7.135034671, -7.298523563, -7.650414989, -7.899712902, -8.137177798, -8.364771275, -8.549384178, -8.695278711, -8.791895682, -8.840137682, -8.852855073, -8.817656857, -8.722449889, -8.585365301, -8.396938329, -8.192421923, -7.9226432, -7.634295158, -7.302860366],
                     [5.705173926, 5.943532285, 5.944564129, 5.945181449, 5.991181971, 6.03538149, 6.070210788, 6.103981958, 6.130119071, 6.163638141, 6.192416314, 6.229329852, 6.262974606, 6.308649854, 6.348209235, 6.405347994, 6.449901934, 6.513259171, 6.566549484, 6.627655852, 6.693567889, 6.75329255, 6.813375678, 6.88227897, 6.955676668, 7.008907354, 7.051845893, 7.113717941, 7.179647455, 7.256398803, 7.30311893, 7.355930579, 7.437194, 7.49032157, 7.569316717, 7.608776141, 7.669796387, 7.735583664, 7.820514659, 7.887670485, 7.952176022, 8.031066972, 8.083085491, 8.133737842, 8.201328387, 8.284776991, 8.333151528, 8.41099061, 8.508934757, 8.572165389, 8.643412382, 8.728673143, 8.822218122, 8.922907783, 9.013175262, 9.113607329, 9.199810431, 9.307795625, 9.375302533, 9.486018941, 9.562181423, 9.64682366, 9.716163498, 9.817178136, 9.905924185, 9.981702948, 10.08643635, 10.19856716, 10.32933112, 10.44459877, 10.58027254, 10.74900086, 10.94261109, 11.20020661, 11.43184127, 11.71231315, 11.94110301, 12.34778248, 12.56710333, 12.87875109, 13.23383847, 13.44505742, 13.88556552, 14.0277597, 14.31316438, 14.49352524, 14.64138774, 14.74977816, 14.79657363, 14.78029135, 14.701774, 14.58253803, 14.37552702, 14.11210289, 13.74190715, 13.33996177, 12.86866361, 12.40715492, 11.84286494, 11.27522954, 10.65363089]])
    
    sig = np.array([[5.252032992, 2.111229525, 2.114873765, 2.11717141, 2.564524618, 3.613991045, 4.95465687, 6.701923158, 8.332347473, 10.75032361, 13.12012104, 16.55072759, 20.00379714, 25.00921056, 29.44034662, 35.62885944, 40.03035795, 45.32035668, 48.70148531, 51.27064788, 52.50143183, 52.38598703, 51.33034809, 49.33488146, 46.75423821, 44.81617047, 43.28115488, 41.14057266, 38.92952694, 36.41276508, 34.93487235, 33.37544066, 31.3868466, 30.46496483, 29.73204125, 29.63784074, 29.77817956, 30.18545606, 30.83022349, 31.25680443, 31.51649885, 31.60223911, 31.5082369, 31.28048103, 30.70944389, 29.46326906, 28.43148037, 26.32546375, 23.14412636, 21.00179075, 18.72131965, 16.41117339, 14.61559547, 13.65181139, 13.60290953, 14.30917491, 15.40295533, 17.20137216, 18.48361868, 20.76169411, 22.44850181, 24.47621442, 26.30143433, 29.29005995, 32.23795538, 34.92685274, 38.71830167, 42.6011536, 46.58667771, 49.46963841, 52.06214501, 54.14340642, 55.1697556, 54.72284996, 52.97654871, 49.64401103, 46.23239178, 39.23648164, 35.19505357, 29.35482393, 22.78798149, 19.01562885, 11.66259772, 9.475469327, 5.421420712, 3.128942803, 1.4445257, 0.373987918, 0.042690275, 0.387257968, 1.376518453, 2.831876825, 5.506742189, 9.324631729, 15.60971435, 23.75212531, 35.12446074, 48.20060947, 66.78640642, 88.29951506, 114.9590749],
                    [1.272643673, 1.989771377, 2.001602492, 2.00875185, 2.725203085, 3.923259514, 5.455048663, 7.659765711, 9.925273533, 13.45031122, 16.82401129, 21.29886468, 25.31081749, 30.46414245, 34.57810128, 39.89779606, 43.52785038, 47.91502281, 50.91168392, 53.58369527, 55.57344311, 56.60555809, 56.94530338, 56.5490137, 55.34075359, 54.06857605, 52.86938771, 50.97847023, 48.8994359, 46.59327326, 45.32214156, 44.04611401, 42.45648132, 41.66767619, 40.83901916, 40.56358194, 40.29030207, 40.15813434, 40.13168018, 40.12459657, 40.03511794, 39.6878752, 39.2492105, 38.62408886, 37.4538169, 35.47307951, 34.0767834, 31.5214167, 27.94193849, 25.53033954, 22.81379425, 19.66830475, 16.46567723, 13.41889369, 11.10661723, 9.045608119, 7.725478132, 6.663482029, 6.328827344, 6.302939206, 6.634186774, 7.293502481, 8.023213021, 9.305666509, 10.55240174, 11.63767504, 13.0875934, 14.50545684, 15.93610581, 16.99208181, 18.00330597, 18.94773484, 19.66243691, 20.1060538, 20.10047057, 19.68212324, 19.06958463, 17.51494244, 16.48373536, 14.84562739, 12.79765766, 11.51519833, 8.742829096, 7.827137247, 5.952800259, 4.71587373, 3.610474491, 2.609908841, 1.822370036, 1.18494324, 0.713218035, 0.410243672, 0.194116776, 0.182087354, 0.487088285, 1.172529927, 2.419044235, 4.102079154, 6.795189422, 10.23119981, 14.85997638],
                    [0.34748478, 0.029050693, 0.02900719, 0.028985126, 0.034268657, 0.045847376, 0.052193613, 0.051893035, 0.048215353, 0.044325171, 0.045751762, 0.055484785, 0.070291244, 0.095478719, 0.119863619, 0.159157615, 0.195817199, 0.265754806, 0.351618316, 0.49761717, 0.738041582, 1.059642009, 1.516348087, 2.253824814, 3.359522352, 4.411947746, 5.435875588, 7.211063439, 9.513803259, 12.73595631, 14.97357318, 17.73917594, 22.44142417, 25.77722456, 31.0700006, 33.84695419, 38.29711656, 43.28724537, 49.99147496, 55.47639376, 60.87504524, 67.6147369, 72.11626769, 76.52313233, 82.40893744, 89.63070094, 93.77307623, 100.3496305, 108.4577998, 113.5995788, 119.3204694, 126.0901969, 133.4603475, 141.3768874, 148.5039524, 156.5175078, 163.5084771, 172.4770502, 178.2388261, 188.0181904, 195.0288045, 203.127657, 210.0114199, 220.4035503, 229.788352, 237.8773899, 248.98505, 260.5778073, 273.5230879, 284.3662552, 296.462574, 310.5898546, 325.716185, 344.3549038, 359.9324256, 377.5834421, 391.169003, 413.8595653, 425.4530478, 441.2886578, 458.5691798, 468.5307966, 488.7314516, 495.1377554, 507.9597861, 516.1794569, 523.2155527, 529.0061062, 532.6676212, 534.3510048, 534.0362773, 532.2656854, 528.3461583, 522.8312946, 514.7137392, 505.7176593, 495.1033398, 484.7236271, 472.1107412, 459.5462609, 445.9542622]])
    
    
    
    
    # This function takes lists of means and variances from multiple models and 
    # calculates the fused mean and variance following the Reification/Fusion approach
    # developed by D. Allaire. This function can handle any number of models.
    yM = np.transpose(np.tile(y, (len(y),1,1)), (2,0,1))
    sigM = np.transpose(np.tile(sig, (len(y),1,1)), (2,0,1))
    
    unoM = np.tile(np.diag(np.ones(len(y))), (yM.shape[0],1,1))
    zeroM = np.abs(unoM-1)
    
    yMT = np.transpose(yM, (0,2,1))
    sigMT = np.transpose(sigM, (0,2,1))
    
    # The following individual value calculations are compacted into the single calculation
    # for alpha, but are left here to aid in understanding of the equation
    
    rho1 = np.divide(np.sqrt(sigM), np.sqrt((yM-yMT)**2 + sigM))
    rho2 = np.sqrt(sigMT)/np.sqrt((yMT-yM)**2 + sigMT)
    rho = (sigMT/(sigMT+sigM))*rho1 + (sigM/(sigMT+sigM))*rho2
    sigma = rho*zeroM*(np.sqrt(sigM*sigMT)) + unoM*sigM
    alpha = np.linalg.inv(sigma)
    
    # alpha = np.linalg.pinv(((sigM/(sigM+sigMT))*np.sqrt(sigM)/np.sqrt((yM-yMT)**2 + sigM) + \
    #     (sigMT/(sigM+sigMT))*np.sqrt(sigMT)/np.sqrt((yMT-yM)**2 + sigMT))*zeroM*(np.sqrt(sigM*sigMT)) + unoM*sigM)
    
    w = (np.sum(alpha,2))/np.tile(np.sum(alpha,axis=(1,2)), (3,1)).transpose()
    
    fused_mean_new = np.sum(w*(y.transpose()), axis=1)
    fused_var_new = 1/np.sum(alpha,axis=(2,1))
    
    
    
    
  
    
    
    
    
    
    
    # fused_mean_new, fused_var_new = reification(mean, var)
    
    fused_mean_old, fused_var_old = reification_old(y, sig)
    
    plt.figure()
    plt.scatter(fused_mean_new, fused_mean_old)
    
    plt.figure()
    plt.scatter(fused_var_new, fused_var_old)
